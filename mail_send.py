# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email(contents: str, to_email: str, from_email: str = "alvin20252528@gmail.com",  subject: str = "SSG CALL BACK") -> None:
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=f"<div><strong>{contents}</strong></div>")
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))  # Changed e.message to str(e) for better compatibility