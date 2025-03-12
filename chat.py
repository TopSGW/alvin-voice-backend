from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import logging
import autogen
from case_details import CaseDetails, insert_case_details
from ai_integration import openai_handler, milvus_handler, schedule_call_back, send_confirmation_email
from mail_send import send_email
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
You are an AI assistant for the Skillsfuture and Workforce Singapore hotline website. Your goals are:

1. Greet users with a standard message: "Hi, thanks for contacting Skillsfuture and Workforce Singapore hotline. Please tell me your inquiry and I will have it recorded and schedule a call back appointment for you."
2. Collect and record case details including the inquiry, person's name, mobile number, and email address.
3. Ask relevant questions.
4. Please schedule a callback appointment with an officer. Request that the user provide an explicit booking time including the year, month, date, and the hour. For example: 'Could you provide the exact date and time, including year, month, day, and hour, so I can schedule the officer's callback accordingly?'
5. Maintain a friendly and professional tone throughout the conversation.

Be adaptive and responsive to the user's needs and interests.
"""

assistant = autogen.AssistantAgent(
    name="SkillsFuture_Assistant",
    system_message=system_prompt,
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_handler.api_key}]},
)

extraction_assistant = autogen.AssistantAgent(
    name="Extraction_Assistant",
    system_message="""You are an AI assistant specialized in extracting specific information from conversations. Your task is to extract the following details from the given conversation:
1. Inquiry: The main question or concern of the user.
2. Name: The user's name.
3. Mobile Number: The user's phone number.
4. Email Address: The user's email address.
5. Appointment Date and Time: This field should reflect the scheduled callback time. If provided, please format the date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). If the user does not supply a complete booking time—including the year, month, day, and hour—this field should be left as an empty string.
Provide the extracted information in a JSON format. If any information is not available, leave it as an empty string.

Example output:
{
    "inquiry": "",
    "name": "",
    "mobile_number": "",
    "email_address": "",
    "appointment_date_time": ""
}
""",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_handler.api_key}]},
)

def validate_date(input_date: datetime | str) -> bool:
    if input_date == "":
        return True

    if input_date is None:
        return True

    # Determine local timezone from current time
    local_tz = datetime.now().astimezone().tzinfo

    # If input_date is naive, assume it's in local time
    if isinstance(input_date, datetime):
        if input_date.tzinfo is None:
            input_date = input_date.replace(tzinfo=local_tz)
        else:
            input_date = input_date.astimezone(local_tz)

    current_datetime = datetime.now(local_tz)

    # Return True if input_date is in the future (ignoring 24-hour window)
    return input_date > current_datetime

def validate_email(email: str) -> bool:
    if email == "":     
        return True
    """
    Returns True if the given email matches a basic validation pattern,
    otherwise returns False.
    """
    import re
    # Regex pattern: one or more allowed characters in the local part,
    # then an '@', then allowed domain characters and a TLD of at least 2 letters.
    pattern = r"^[\w\.\+\-]+@[\w\.\-]+\.[a-zA-Z]{2,}$"
    return re.fullmatch(pattern, email) is not None

async def chat(conversation_request: ConversationRequest) -> ConversationResponse:
    try:
        # Prepare the conversation history
        conversation = [{"role": msg.role, "content": msg.content} for msg in conversation_request.conversation_history]
        conversation.append({"role": "user", "content": conversation_request.user_input})

        # Generate response using the AssistantAgent
        logger.debug("Generating reply using AssistantAgent")
        response = assistant.generate_reply(conversation)
        ai_response = response
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
            embedding = openai_handler.emb_text(case_details.inquiry)
            search_result = milvus_handler.search(embedding)
            if search_result:
                case_details.category_text = search_result[0][0]['entity']['text']
                case_details.divide_text = search_result[0][0]['entity']['divide_text']
            if insert_case_details(case_details):
                schedule_call_back(case_details.appointment_date_time)
                send_confirmation_email(case_details.email_address, case_details.appointment_date_time)
        
        return ConversationResponse(
            ai_response=ai_response,
            updated_history=updated_history,
            case_details=case_details
        )
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def extract_case_details(conversation_history: List[Message]) -> CaseDetails:
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
    
    extraction_result = extraction_assistant.generate_reply(messages=[{"role": "user", "content": extraction_prompt}])
    logger.debug(f"Raw extraction result: {extraction_result}")
    
    extraction_content = extraction_result[1] if isinstance(extraction_result, tuple) and len(extraction_result) > 1 else extraction_result
    logger.debug(f"Extraction content: {extraction_content}")
    
    try:
        if not extraction_content or not extraction_content.strip():
            logger.error("Extraction result is empty")
            return CaseDetails()
        
        # Try to find JSON content within the extraction result
        json_start = extraction_content.find('{')
        json_end = extraction_content.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_content = extraction_content[json_start:json_end]
            logger.debug(f"Extracted JSON content: {json_content}")
            extracted_data = json.loads(json_content)
        else:
            logger.error("No valid JSON found in extraction content")
            return CaseDetails()
        
        # Parse appointment_date_time if it's not empty
        if extracted_data["appointment_date_time"]:
            try:
                appointment_datetime = datetime.fromisoformat(extracted_data["appointment_date_time"])
                extracted_data["appointment_date_time"] = appointment_datetime
            except ValueError:
                logger.warning("Invalid appointment_date_time format. Setting to None.")
                extracted_data["appointment_date_time"] = None
        else:
            extracted_data["appointment_date_time"] = None
        
        logger.info(f"Successfully extracted case details: {extracted_data}")
        return CaseDetails(**extracted_data)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {str(e)}")
        logger.error(f"Invalid JSON content: {extraction_content}")
        return CaseDetails()
    except ValueError as e:
        logger.error(f"Error processing extraction result: {str(e)}")
        return CaseDetails()