from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from autogen import ConversableAgent
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from openai import OpenAI

# Load environment variables
load_dotenv()

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
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

config_list = [{"model": "gpt-4", "api_key": openai_api_key}]

# Database connection
DB_NAME = os.environ.get("DB_NAME", "voicebot")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

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
        category_text TEXT,
        divide_text TEXT,
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
    category_text: str = ""
    divide_text: str = ""

class ConversationResponse(BaseModel):
    ai_response: str
    updated_history: List[Message]
    case_details: CaseDetails

class MilvusHandler:
    def __init__(self):
        self.milvus_client = MilvusClient("./milvus_demo.db")
        self.collection_name = "alvin_collection"
        self.setup_collection()

    def setup_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="divide_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        schema = CollectionSchema(fields, "Collection for storing text + embeddings")

        if self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.drop_collection(collection_name=self.collection_name)

        self.milvus_client.create_collection(
            collection_name=self.collection_name, 
            schema=schema, 
            metric_type='IP'
        )

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.milvus_client.release_collection(collection_name=self.collection_name)
        self.milvus_client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="FLAT", 
            metric_type="IP", 
            params={
                # "M": 16,
                # "efConstruction": 500,
            },
        )
        self.milvus_client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def insert_data(self, vector_data):
        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=vector_data
        )

    def search(self, query_vector, limit=1):
        search_params = {
            "metric_type": "IP",
            "params": {}
        }
        return self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text", "divide_text"],
            search_params=search_params
        )

class OpenAIHandler:
    def __init__(self):
        self.client = OpenAI(api_key=openai_api_key)

    def emb_text(self, text):
        return self.client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

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

# Create the assistant agent with built-in memory
assistant = ConversableAgent(
    name="SkillsFuture_Assistant",
    system_message=system_prompt,
    llm_config={"config_list": config_list},
    human_input_mode="NEVER"
)

extraction_assistant = ConversableAgent(
    name="Extraction_Assistant",
    system_message="""You are an AI assistant specialized in extracting specific information from conversations. Your task is to extract the following details from the given conversation:
1. Inquiry: The main question or concern of the user.
2. Name: The user's name.
3. Mobile Number: The user's phone number.
4. Email Address: The user's email address.
5. Appointment Date and Time: The scheduled callback time. Please provide this in ISO format (YYYY-MM-DDTHH:MM:SS) if available, otherwise leave it as an empty string.

Provide the extracted information in a JSON format. If any information is not available, leave it as an empty string.""",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER"
)

milvus_handler = MilvusHandler()
openai_handler = OpenAIHandler()

def load_category_and_divide_text():
    with open("category.txt", "r", encoding="utf-8") as f:
        category_contents = [line.strip() for line in f if line.strip()]
    
    with open("divids.txt", "r", encoding="utf-8") as f:
        divide_contents = [line.strip() for line in f if line.strip()]
    
    vector_data = []
    for content, divide_text in zip(category_contents, divide_contents):
        embedding_val = openai_handler.emb_text(content)
        vector_data.append({
            "text": content,
            "vector": embedding_val,
            "divide_text": divide_text
        })
    
    milvus_handler.insert_data(vector_data)
    logger.info(f"Loaded {len(vector_data)} items into Milvus")

# Load category and divide text data into Milvus when the application starts
# load_category_and_divide_text()

def insert_case_details(case_details: CaseDetails):
    if not all([case_details.inquiry, case_details.name, case_details.mobile_number, case_details.email_address, case_details.appointment_date_time]):
        logger.info("Skipping case insertion due to incomplete information")
        return False

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO case_details (inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        case_details.inquiry,
        case_details.name,
        case_details.mobile_number,
        case_details.email_address,
        case_details.appointment_date_time,
        case_details.category_text,
        case_details.divide_text
    ))
    conn.commit()
    cur.close()
    conn.close()
    return True

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    try:
        # Prepare the conversation history
        conversation = [{"role": "user", "content": request.user_input}]
        
        # Generate response using the AssistantAgent
        response = assistant.generate_reply(messages=conversation)
        ai_response = response[1].strip() if isinstance(response, tuple) and len(response) > 1 else ""
        
        # Limit response length if needed
        max_response_length = 500  # Adjust as needed
        if len(ai_response) > max_response_length:
            ai_response = ai_response[:max_response_length].rsplit(' ', 1)[0] + '...'
        
        # Update conversation history
        updated_history = request.conversation_history + [
            Message(role="user", content=request.user_input),
            Message(role="assistant", content=ai_response)
        ]
        
        case_details = extract_case_details(updated_history)
        
        # Get category and divide text using Milvus
        embedding = openai_handler.emb_text(case_details.inquiry)
        search_result = milvus_handler.search(embedding)
        if search_result:
            case_details.category_text = search_result[0][0]['entity']['text']
            case_details.divide_text = search_result[0][0]['entity']['divide_text']
        
        # Insert case details if all fields are non-empty
        if all([case_details.inquiry, case_details.name, case_details.mobile_number, case_details.email_address, case_details.appointment_date_time]):
            if insert_case_details(case_details):
                schedule_call_back(case_details.appointment_date_time)
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
    
    extraction_result = extraction_assistant.generate_reply(messages=[{"role": "user", "content": extraction_prompt}])
    extraction_content = extraction_result[1] if isinstance(extraction_result, tuple) and len(extraction_result) > 1 else ""
    
    try:
        extracted_data = json.loads(extraction_content)
        if extracted_data["appointment_date_time"]:
            extracted_data["appointment_date_time"] = datetime.fromisoformat(extracted_data["appointment_date_time"])
        else:
            extracted_data["appointment_date_time"] = None
        return CaseDetails(**extracted_data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error processing extraction result: {str(e)}")
        return CaseDetails()

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