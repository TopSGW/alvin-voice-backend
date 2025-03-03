from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import autogen
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

config_list = [{"model": "gpt-4", "api_key": openai_api_key}]

# Database connection
DB_NAME = os.getenv("DB_NAME", "voicebot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# Create table if not exists
def create_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS case_details (
        id SERIAL PRIMARY KEY,
        inquiry TEXT,
        name VARCHAR(100),
        mobile_number VARCHAR(20),
        email_address VARCHAR(100),
        appointment_date_time TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    cur.close()
    conn.close()

create_table()

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    user_input: str
    conversation_history: List[Message] = []

class CaseDetails(BaseModel):
    inquiry: str = ""
    name: str = ""
    mobile_number: str = ""
    email_address: str = ""
    appointment_date_time: Optional[datetime] = None

class ConversationResponse(BaseModel):
    ai_response: str
    updated_history: List[Message]
    case_details: CaseDetails

system_prompt = """
You are an AI assistant for the Skillsfuture and Workforce Singapore hotline website. Your goals are:

1. Greet users with a standard message: "Hi, thanks for contacting Skillsfuture and Workforce Singapore hotline. Please tell me your inquiry and I will have it recorded and schedule a call back appointment for you."
2. Collect and record case details including the inquiry, person's name, mobile number, and email address.
3. Ask relevant questions to gather more information about the user's background and needs.
4. Provide information about Skillsfuture credits and suitable courses based on the user's background.
5. Please schedule a callback appointment with an officer. Request that the user provide an explicit booking time including the year, month, date, and the hour. For example: 'Could you provide the exact date and time, including year, month, day, and hour, so I can schedule the officer's callback accordingly?'
6. Maintain a friendly and professional tone throughout the conversation.

Be adaptive and responsive to the user's needs and interests.
"""

assistant = autogen.AssistantAgent(
    name="SkillsFuture_Assistant",
    system_message=system_prompt,
    llm_config={"config_list": config_list},
)

extraction_assistant = autogen.AssistantAgent(
    name="Extraction_Assistant",
    system_message="""You are an AI assistant specialized in extracting specific information from conversations. Your task is to extract the following details from the given conversation:
1. Inquiry: The main question or concern of the user.
2. Name: The user's name.
3. Mobile Number: The user's phone number.
4. Email Address: The user's email address.
5. Appointment Date and Time: The scheduled callback time. Please provide this in ISO format (YYYY-MM-DDTHH:MM:SS) if available, otherwise leave it as an empty string.

Provide the extracted information in a JSON format. If any information is not available, leave it as an empty string.""",
    llm_config={"config_list": config_list},
)

def insert_case_details(case_details: CaseDetails):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO case_details (inquiry, name, mobile_number, email_address, appointment_date_time)
    VALUES (%s, %s, %s, %s, %s)
    """, (
        case_details.inquiry,
        case_details.name,
        case_details.mobile_number,
        case_details.email_address,
        case_details.appointment_date_time
    ))
    conn.commit()
    cur.close()
    conn.close()

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    try:
        conversation = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        conversation.append({"role": "user", "content": request.user_input})
        
        # Generate response using the AssistantAgent
        logger.debug("Generating reply using AssistantAgent")
        response = assistant.generate_reply(conversation)
        ai_response = response
        logger.debug(f"AssistantAgent response: {ai_response}")
        
        updated_history = request.conversation_history + [
            Message(role="user", content=request.user_input),
            Message(role="assistant", content=ai_response)
        ]
        
        logger.debug("Extracting case details")
        case_details = extract_case_details(updated_history)
        logger.debug(f"Extracted case details: {case_details}")
        
        # Insert case details into the database
        logger.debug("Inserting case details into database")
        insert_case_details(case_details)
        
        if case_details.appointment_date_time:
            logger.debug(f"Scheduling call back for {case_details.appointment_date_time}")
            schedule_call_back(case_details.appointment_date_time)
            if case_details.email_address:
                logger.debug(f"Sending confirmation email to {case_details.email_address}")
                send_confirmation_email(case_details.email_address, case_details.appointment_date_time)
        
        return ConversationResponse(
            ai_response=ai_response,
            updated_history=updated_history,
            case_details=case_details
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
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
    
    logger.debug("Generating reply for extraction")
    extraction_result = extraction_assistant.generate_reply([{"role": "user", "content": extraction_prompt}])
    logger.debug(f"Extraction result: {extraction_result}")
    
    try:
        extracted_data = json.loads(extraction_result)
        # Convert appointment_date_time to datetime object if it's not empty
        if extracted_data["appointment_date_time"]:
            extracted_data["appointment_date_time"] = datetime.fromisoformat(extracted_data["appointment_date_time"])
        else:
            extracted_data["appointment_date_time"] = None
        return CaseDetails(**extracted_data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        logger.error(f"Problematic JSON string: {extraction_result}")
        # If JSON parsing fails, return an empty CaseDetails object
        return CaseDetails()
    except ValueError as e:
        logger.error(f"Date parsing error: {str(e)}")
        logger.error(f"Problematic date string: {extracted_data.get('appointment_date_time')}")
        # If date parsing fails, set appointment_date_time to None
        extracted_data["appointment_date_time"] = None
        return CaseDetails(**extracted_data)

def schedule_call_back(appointment_date_time: datetime) -> bool:
    logger.info(f"Call back scheduled for: {appointment_date_time}")
    return True

def send_confirmation_email(email_address: str, appointment_date_time: datetime) -> bool:
    logger.info(f"Confirmation email sent to {email_address} for appointment at {appointment_date_time}")
    return True

@app.get("/case_details")
async def get_case_details():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM case_details ORDER BY created_at DESC")
    case_details = cur.fetchall()
    cur.close()
    conn.close()
    return {"case_details": case_details}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)