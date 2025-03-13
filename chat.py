from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import logging
import asyncio
from case_details import CaseDetails, insert_case_details
from ai_integration import openai_handler, milvus_handler, schedule_call_back, send_confirmation_email, OpenAIHandler
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    user_input: str
    conversation_history: List[Message] = []

class ConversationResponse(BaseModel):
    ai_response: str
    updated_history: List[Message]
    case_details: CaseDetails

# Initialize AI assistants
system_prompt = """
You are an AI guidance helper bot for the Skillsfuture and Workforce Singapore hotline. Your goals are:

1. Always greet users with: "Hi, thanks for contacting Skillsfuture and Workforce Singapore. I am an automated call scheduler bot. Please tell me your inquiry and I will have it recorded."
2. Before collecting the user's details (inquiry, name, mobile number, and email address), first confirm the nature of their inquiry:
   - If the inquiry is about a course, ask for the course name and training provider.
   - If the inquiry is about job hunting, ask about the user's employment status.
   - If the inquiry is about a grant, ask for the date of submission.
   - If the inquiry involves using SkillsFuture credit, confirm the user's citizenship.
3. Collect and record case details (inquiry, name, mobile number, and email address) after clarifying step #2.
4. Ask only one question in each response, keeping the conversation short and clear.
5. Schedule a callback appointment by requesting an explicit booking time (assuming the year 2025). For example:
   “Could you provide the exact date and time, including the year (2025), month, day, and hour, so I can schedule the officer’s callback accordingly?”
6. Maintain a friendly and professional tone.
7. Be adaptive and responsive to the user's needs, without asking multiple questions at once.
"""

assistant = OpenAIHandler(system_prompt=system_prompt)

extraction_assistant = OpenAIHandler(
    system_prompt="""
You are an AI assistant specialized in extracting specific information from conversations. Your tasks are:

1. Analyze all inquiry-related details and generate a clear, concise guidance sentence that summarizes:
   - Who is inquiring (the user’s name).
   - The subject of the inquiry (e.g., SkillsFuture credit, job hunting, grant, course name, training provider, citizenship, etc.).
   - Any additional context the user provides.
   
   For example:
   "Mr Alvin Lim inquired about his use of SkillsFuture credits for IT Business Analytics from SMU. He is a Singapore citizen. Please advise him if he could do so."

2. Store this guidance sentence in the "inquiry" field of your JSON output.

3. Extract and provide the following details from the conversation:
   - Name: The user's name.
   - Mobile Number: The user's phone number.
   - Email Address: The user's email address.
   - Appointment Date and Time: This field should reflect the scheduled callback time. If provided, please format the date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). If the user does not supply a complete booking time—including the year, month, day, and hour—leave this field as an empty string.

If any information is not available, leave it as an empty string.

Your output must be valid JSON. For example:

{
  "inquiry": "",
  "name": "",
  "mobile_number": "",
  "email_address": "",
  "appointment_date_time": ""
}
""",
)

def validate_date(input_date: Optional[datetime]) -> bool:
    if not input_date:
        return True
    return input_date.replace(tzinfo=None) > datetime.now()

def validate_email(email: str) -> bool:
    import re
    pattern = r"^[\w\.\+\-]+@[\w\.\-]+\.[a-zA-Z]{2,}$"
    return email == "" or re.fullmatch(pattern, email) is not None

async def chat(conversation_request: ConversationRequest) -> ConversationResponse:
    try:
        # Prepare the conversation history
        conversation = [{"role": msg.role, "content": msg.content} for msg in conversation_request.conversation_history]
        conversation.append({"role": "user", "content": conversation_request.user_input})

        # Generate response using the AssistantAgent
        logger.debug("Generating reply using AssistantAgent")
        ai_response = await assistant.agenerate_chat_completion(conversation)
        logger.debug(f"AssistantAgent response: {ai_response}")
        
        # Update conversation history
        updated_history = conversation_request.conversation_history + [
            Message(role="user", content=conversation_request.user_input),
            Message(role="assistant", content=ai_response)
        ]
        
        case_details = extract_case_details(updated_history)

        logger.info(f"Extracted case details: {case_details}")

        # Check if the appointment date is in the past
        if not validate_date(case_details.appointment_date_time):
            logger.warning(f"Invalid appointment date: {case_details.appointment_date_time}")
            ai_response = "\n\nI apologize, but it seems the appointment date you provided is in the past. Could you please provide a future date and time for the appointment?"   
            return ConversationResponse(
                ai_response=ai_response,
                updated_history=updated_history,
                case_details=case_details
            )

        if not validate_email(case_details.email_address):
            logger.warning(f"Invalid email address: {case_details.email_address}")
            ai_response = "The email address you entered appears to be invalid. Please provide a valid email address."
            return ConversationResponse(
                ai_response=ai_response,
                updated_history=updated_history,
                case_details=case_details
            )

        # Insert case details if all fields are non-empty
        if all([case_details.inquiry, case_details.name, case_details.mobile_number, case_details.email_address, case_details.appointment_date_time]):
            # Get category and divide text using Milvus
            embedding = await openai_handler.aemb_text(case_details.inquiry)
            search_result = milvus_handler.search(embedding)
            if search_result:
                case_details.category_text = search_result[0][0]['entity']['text']
                case_details.divide_text = search_result[0][0]['entity']['divide_text']
            if insert_case_details(case_details):
                await asyncio.gather(
                    schedule_call_back(case_details.appointment_date_time),
                    send_confirmation_email(case_details.email_address, case_details.appointment_date_time)
                )
        
        return ConversationResponse(
            ai_response=ai_response,
            updated_history=updated_history,
            case_details=case_details
        )
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def extract_case_details(conversation_history: List[Message]) -> CaseDetails:
    conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation_history])
    extraction_prompt = f"""Please extract the case details from the following conversation:

{conversation_text}

Provide the extracted information in the following JSON format:
{{
    "inquiry": "Extracted Inquiry",
    "name": "Extracted Name",
    "mobile_number": "Extracted Mobile Number",
    "email_address": "Extracted Email Address",
    "appointment_date_time": "Extracted Appointment Date and Time in ISO format (YYYY-MM-DDTHH:MM:SS) or empty string if not available"
}}
If any information is not available, leave it as an empty string."""
    
    logger.debug(f"Extraction prompt: {extraction_prompt}")
    
    extraction_result = await extraction_assistant.agenerate_chat_completion(messages=[{"role": "user", "content": extraction_prompt}])
    try:
        data = json.loads(extraction_result)
        appointment = data.get("appointment_date_time", "")
        if appointment:
            data["appointment_date_time"] = datetime.fromisoformat(appointment)
        else:
            data["appointment_date_time"] = None
        return CaseDetails(**data)
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return CaseDetails()