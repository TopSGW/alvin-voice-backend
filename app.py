from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from autogen import ConversableAgent
import autogen
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime, timezone, timedelta
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from openai import OpenAI
import re
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(100) UNIQUE NOT NULL,
        hashed_password VARCHAR(100) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    cur.close()
    conn.close()

create_table()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    user_input: str
    conversation_history: List[Message] = []

class CaseDetails(BaseModel):
    id: Optional[int] = None
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

class User(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class ChangeCredentialRequest(BaseModel):
    new_email: EmailStr
    new_password: str = Field(..., min_length=8)

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
            print("collection is existing!")
        else: 
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
                params={},
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
3. Ask relevant questions.
4. Please schedule a callback appointment with an officer. Request that the user provide an explicit booking time including the year, month, date, and the hour. For example: 'Could you provide the exact date and time, including year, month, day, and hour, so I can schedule the officer's callback accordingly?'
5. Maintain a friendly and professional tone throughout the conversation.

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
    llm_config={"config_list": config_list},
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

def validate_date(input_date: datetime | str) -> bool:
    if input_date == "":
        return True

    if input_date is None:
        return True

    # Determine local timezone from current time
    local_tz = datetime.now().astimezone().tzinfo

    # If input_date is naive, assume it's in local time
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
    # Regex pattern: one or more allowed characters in the local part,
    # then an '@', then allowed domain characters and a TLD of at least 2 letters.
    pattern = r"^[\w\.\+\-]+@[\w\.\-]+\.[a-zA-Z]{2,}$"
    return re.fullmatch(pattern, email) is not None

# User authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(email: str):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user['hashed_password']):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# Set default credential
DEFAULT_EMAIL = "alvin2525@gmail.com"
DEFAULT_PASSWORD = "alvin123456"

def set_default_credential():
    user = get_user(DEFAULT_EMAIL)
    if not user:
        hashed_password = get_password_hash(DEFAULT_PASSWORD)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (email, hashed_password) VALUES (%s, %s)", (DEFAULT_EMAIL, hashed_password))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Default credential set for {DEFAULT_EMAIL}")
    else:
        logger.info(f"Default credential already exists for {DEFAULT_EMAIL}")

# Call this function when the app starts
set_default_credential()

@app.post("/chat", response_model=ConversationResponse)
@limiter.limit("5/minute")
async def chat(request: Request, conversation_request: ConversationRequest):
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

        print("case details : ", case_details)        
        
        print("type >>>>>>>>>>>>>>>>", type(case_details.email_address))

        # Check if the appointment date is in the past
        if(validate_date(case_details.appointment_date_time) == False):
            print("fffffffffffffffddddddddddate", case_details.appointment_date_time)
            ai_response = "\n\nI apologize, but it seems the appointment date you provided is in the past. Could you please provide a future date and time for the appointment?"   
            return ConversationResponse(
                ai_response=ai_response,
                updated_history=updated_history,
                case_details=case_details
            )

        if(validate_email(case_details.email_address) == False):
            print ("email address >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Invalide <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
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
            ai_response = ai_response + "\n" + "tag: " + case_details.category_text + "\n" + "division: " + case_details.divide_text + "\n"
            print("ai response: ", ai_response)
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

def schedule_call_back(appointment_date_time: datetime) -> bool:
    logger.info(f"Call back scheduled for: {appointment_date_time}")
    return True

def send_confirmation_email(email_address: str, appointment_date_time: datetime) -> bool:
    logger.info(f"Confirmation email sent to {email_address} for appointment at {appointment_date_time}")
    return True

@app.get("/case_details")
async def get_case_details(current_user: User = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM case_details ORDER BY created_at DESC")
    case_details = cur.fetchall()
    cur.close()
    conn.close()
    return {"case_details": case_details}

# Create operation
@app.post("/case_details", response_model=CaseDetails)
async def create_case_detail(case_detail: CaseDetails, current_user: User = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
        INSERT INTO case_details (inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text
        """, (
            case_detail.inquiry,
            case_detail.name,
            case_detail.mobile_number,
            case_detail.email_address,
            case_detail.appointment_date_time,
            case_detail.category_text,
            case_detail.divide_text
        ))
        new_case_detail = cur.fetchone()
        conn.commit()
        return CaseDetails(**new_case_detail)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# Read operation for a single case detail
@app.get("/case_details/{case_id}", response_model=CaseDetails)
async def read_case_detail(case_id: int, current_user: User = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM case_details WHERE id = %s", (case_id,))
        case_detail = cur.fetchone()
        if case_detail is None:
            raise HTTPException(status_code=404, detail="Case detail not found")
        return CaseDetails(**case_detail)
    finally:
        cur.close()
        conn.close()

# Update operation
@app.put("/case_details/{case_id}", response_model=CaseDetails)
async def update_case_detail(case_id: int, case_detail: CaseDetails, current_user: User = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
        UPDATE case_details
        SET inquiry = %s, name = %s, mobile_number = %s, email_address = %s, 
            appointment_date_time = %s, category_text = %s, divide_text = %s
        WHERE id = %s
        RETURNING id, inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text
        """, (
            case_detail.inquiry,
            case_detail.name,
            case_detail.mobile_number,
            case_detail.email_address,
            case_detail.appointment_date_time,
            case_detail.category_text,
            case_detail.divide_text,
            case_id
        ))
        updated_case_detail = cur.fetchone()
        if updated_case_detail is None:
            raise HTTPException(status_code=404, detail="Case detail not found")
        conn.commit()
        return CaseDetails(**updated_case_detail)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# Delete operation
@app.delete("/case_details/{case_id}", response_model=dict)
async def delete_case_detail(case_id: int, current_user: User = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM case_details WHERE id = %s RETURNING id", (case_id,))
        deleted_case = cur.fetchone()
        if deleted_case is None:
            raise HTTPException(status_code=404, detail="Case detail not found")
        conn.commit()
        return {"message": f"Case detail with id {case_id} has been deleted"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# Read all operation with pagination
@app.get("/case_details", response_model=List[CaseDetails])
async def read_case_details(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM case_details ORDER BY created_at DESC OFFSET %s LIMIT %s", (skip, limit))
        case_details = cur.fetchall()
        return [CaseDetails(**case) for case in case_details]
    finally:
        cur.close()
        conn.close()


@app.post("/register", response_model=Token)
@limiter.limit("5/minute")
async def register_user(request: Request, user: User):
    db_user = get_user(user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (email, hashed_password) VALUES (%s, %s)", (user.email, hashed_password))
    conn.commit()
    cur.close()
    conn.close()
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
@limiter.limit("5/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['email']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    # In a stateless JWT-based authentication system, we can't invalidate the token on the server side.
    # Instead, we'll return a success message and the client should remove the token from local storage.
    return {"message": "Successfully logged out"}

@app.post("/change_credential")
async def change_credential(request: ChangeCredentialRequest, current_user: User = Depends(get_current_user)):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        hashed_password = get_password_hash(request.new_password)
        cur.execute("""
        UPDATE users
        SET email = %s, hashed_password = %s
        WHERE email = %s
        """, (request.new_email, hashed_password, current_user['email']))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()
        return {"message": "Credential updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)