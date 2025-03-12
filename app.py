from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from datetime import timedelta
from typing import List

from auth import (
    User, Token, ChangeCredentialRequest, authenticate_user, create_access_token,
    get_current_user, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, set_default_credential,
    get_user  # Add this import
)
from case_details import (
    CaseDetails, get_case_details, create_case_detail, read_case_detail,
    update_case_detail, delete_case_detail, read_case_details
)
from chat import chat, ConversationRequest, ConversationResponse
from database import create_tables, get_db_connection
from ai_integration import load_category_and_divide_text

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

# Create tables when the app starts
create_tables()

# Set default credential when the app starts
DEFAULT_EMAIL = "alvin2525@gmail.com"
DEFAULT_PASSWORD = "alvin123456"
set_default_credential(email=DEFAULT_EMAIL, password=DEFAULT_PASSWORD)

# Load category and divide text data into Milvus when the application starts
# Uncomment the following line when ready to load data into Milvus
# load_category_and_divide_text()

@app.post("/chat", response_model=ConversationResponse)
@limiter.limit("5/minute")
async def chat_endpoint(request: Request, conversation_request: ConversationRequest):
    return await chat(conversation_request)

@app.get("/case_details", response_model=List[CaseDetails])
async def get_case_details_endpoint(current_user: User = Depends(get_current_user)):
    case_details = await get_case_details()
    return case_details['case_details'] if isinstance(case_details, dict) else case_details

@app.post("/case_details", response_model=CaseDetails)
async def create_case_detail_endpoint(case_detail: CaseDetails, current_user: User = Depends(get_current_user)):
    return await create_case_detail(case_detail)

@app.get("/case_details/{case_id}", response_model=CaseDetails)
async def read_case_detail_endpoint(case_id: int, current_user: User = Depends(get_current_user)):
    return await read_case_detail(case_id)

@app.put("/case_details/{case_id}", response_model=CaseDetails)
async def update_case_detail_endpoint(case_id: int, case_detail: CaseDetails, current_user: User = Depends(get_current_user)):
    return await update_case_detail(case_id, case_detail)

@app.delete("/case_details/{case_id}")
async def delete_case_detail_endpoint(case_id: int, current_user: User = Depends(get_current_user)):
    return await delete_case_detail(case_id)

@app.get("/case_details", response_model=List[CaseDetails])
async def read_case_details_endpoint(skip: int = 0, limit: int = 10, current_user: User = Depends(get_current_user)):
    return await read_case_details(skip, limit)

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
        data={"sub": user.email}, expires_delta=access_token_expires
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
        """, (request.new_email, hashed_password, current_user.email))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()
        return {"message": "Credentials updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)