import os
from dotenv import load_dotenv
from openai import OpenAI
import logging
from milvus_manager import MilvusHandler
from datetime import datetime
from mail_send import send_email
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenAIHandler:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=self.api_key)

    def emb_text(self, text):
        return self.client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

# Initialize OpenAI and Milvus handlers
openai_handler = OpenAIHandler()
milvus_handler = MilvusHandler()

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

# Call this function when the application starts
# load_category_and_divide_text()

def ordinal(n: int) -> str:
    """Return the ordinal string of an integer (e.g., 1 -> '1st', 2 -> '2nd')."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def schedule_call_back(appointment_date_time: datetime) -> bool:
    # This is a placeholder function. In a real-world scenario, you would implement
    # the actual scheduling logic here, possibly integrating with a calendar system.
    logger.info(f"Call back scheduled for: {appointment_date_time}")
    return True

def send_confirmation_email(email_address: str, appointment_date_time: datetime) -> bool:
    # This is a placeholder function. In a real-world scenario, you would implement
    formatted_date_time = (
        f"{appointment_date_time.strftime('%B')} "  # Full month name, e.g., May
        f"{ordinal(appointment_date_time.day)}, "    # Day with ordinal suffix, e.g., 1st
        f"{appointment_date_time.year} at "           # Year and literal 'at'
        f"{appointment_date_time.hour % 12 or 12} "     # 12-hour format hour without leading zero
        f"{'AM' if appointment_date_time.hour < 12 else 'PM'}"  # AM/PM
    )
    
    content = (
        "Dear Customer,\n\n"
        f"Thank you for scheduling a callback with our officer. We have arranged a callback on {formatted_date_time}.\n\n"
        "If you have any questions or need to reschedule, please let us know.\n\n"
        "Best regards,\n"
    )
    
    # The actual email sending logic goes here.
    send_email(to_email=email_address, content=content)

    logger.info(f"Confirmation email sent to {email_address} for appointment at {appointment_date_time}")
    return True