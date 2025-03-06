# SkillsFuture and Workforce Singapore Hotline AI Assistant

## Project Overview

This project implements an AI-powered assistant for the SkillsFuture and Workforce Singapore hotline website. The assistant is designed to handle user inquiries, collect case details, provide information about SkillsFuture credits and courses, and schedule callback appointments with officers.

## Technology Stack

- **Backend Framework**: FastAPI
- **Database**: PostgreSQL
- **AI and NLP**: OpenAI GPT-4
- **Vector Database**: Milvus
- **Conversation Management**: Autogen
- **Python Version**: 3.9+ (inferred from dependencies)

## Main Components

### 1. FastAPI Application (app.py)

The main application file that sets up the FastAPI server, defines API endpoints, and orchestrates the various components of the system.

### 2. Database Handler

- Manages connections to the PostgreSQL database
- Creates and interacts with the `case_details` table

### 3. MilvusHandler

- Manages interactions with the Milvus vector database
- Used for storing and searching category and divide text embeddings

### 4. OpenAIHandler

- Handles interactions with the OpenAI API
- Generates text embeddings for similarity search

### 5. Autogen Agents

- SkillsFuture_Assistant: Main conversational agent
- Extraction_Assistant: Specialized agent for extracting case details from conversations

## API Endpoints

1. POST `/chat`
   - Handles user input and generates AI responses
   - Extracts case details and stores them in the database
   - Performs similarity search for category and divide text

2. GET `/case_details`
   - Retrieves all case details from the database

## Database Structure

Table: `case_details`

Columns:
- id (SERIAL PRIMARY KEY)
- inquiry (TEXT)
- name (VARCHAR(100))
- mobile_number (VARCHAR(20))
- email_address (VARCHAR(100))
- appointment_date_time (TIMESTAMP)
- category_text (TEXT)
- divide_text (TEXT)
- created_at (TIMESTAMP)

## External Integrations

1. OpenAI API
   - Used for generating conversational responses and text embeddings

2. Milvus Vector Database
   - Used for storing and searching category and divide text embeddings

## Setup and Deployment

1. Environment Setup
   - Create a `.env` file with the following variables:
     - OPENAI_API_KEY
     - DB_NAME
     - DB_USER
     - DB_PASSWORD
     - DB_HOST
     - DB_PORT

2. Install Dependencies
   - Run `pip install -r requirements.txt` to install all required packages

3. Database Setup
   - Ensure PostgreSQL is installed and running
   - The application will automatically create the necessary table on startup

4. Milvus Setup
   - Ensure Milvus is installed and running
   - The application will set up the necessary collection on startup

5. Running the Application
   - Execute `uvicorn app:app --host 0.0.0.0 --port 8000` to start the server
   - The API will be available at `http://localhost:8000`

## Key Functionalities

1. Conversational AI
   - Greets users and collects case details
   - Provides information about SkillsFuture credits and courses
   - Schedules callback appointments

2. Case Detail Extraction
   - Automatically extracts relevant information from user conversations

3. Similarity Search
   - Uses Milvus to find relevant category and divide text based on user inquiries

4. Database Integration
   - Stores and retrieves case details from PostgreSQL

5. Email Confirmation
   - Sends confirmation emails for scheduled appointments (simulated in the current implementation)

## Security Considerations

- CORS is configured to allow all origins (`"*"`). In a production environment, this should be restricted to specific allowed origins.
- API keys and database credentials are stored in environment variables for security.
- Ensure proper security measures are implemented when deploying to a production environment, including HTTPS, rate limiting, and proper authentication for sensitive endpoints.

This technical documentation provides an overview of the SkillsFuture and Workforce Singapore Hotline AI Assistant project. For more detailed information on specific components or functionalities, refer to the inline comments in the source code or consult the respective library documentations.