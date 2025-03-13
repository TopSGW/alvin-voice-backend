import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from datetime import datetime

def ordinal(n: int) -> str:
    """Return the ordinal string of an integer (e.g., 1 -> '1st', 2 -> '2nd')."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

async def send_email(
    appointment_date_time: datetime,
    to_email: str,
    from_email: str = "alvin20252528@gmail.com",
    subject: str = "Call back scheduled with SkillsFuture and Workforce Singapore",
    phone_number: str = "6785 5785"
) -> None:
    """
    Sends an email using SendGrid with the SSG-WSG style shown in the screenshot.
    
    :param appointment_date_time: The datetime object for the scheduled call.
    :param to_email: Recipient's email address.
    :param from_email: Sender's email address (default provided).
    :param subject: Email subject (defaults to "Call back scheduled with SkillsFuture and Workforce Singapore").
    :param phone_number: The callback phone number to display in the email body.
    """

    # Format the date and time with an ordinal day (e.g., "14th March 2025 at 2 PM").
    formatted_date_time = (
        f"{appointment_date_time.strftime('%d')}<sup>{ordinal(appointment_date_time.day)[-2:]}</sup> "
        f"{appointment_date_time.strftime('%B %Y')} at "
        f"{appointment_date_time.strftime('%-I %p')}"
    )

    # Build the HTML content to match the style in your screenshot.
    html_content = f"""
    <p><strong>Subject:</strong> Call back scheduled with <em>SkillsFuture</em> and <em>Workforce Singapore</em></p>
    <p>Dear Sir/Mdm,</p>

    <p>
      Thank you for scheduling a callback with our officer. 
      Your call appointment is scheduled to be on 
      <strong>{formatted_date_time}</strong>. 
      Please note that the number that will be calling you will be <strong>{phone_number}</strong>.
    </p>

    <p>
      If you have any questions or need to reschedule, 
      please use <strong>SSG-WSG CallMeBack</strong> call scheduler bot again.
    </p>

    <p>Thank you.</p>

    <p>
      <strong>SkillsFuture Singapore</strong><br>
      <strong>Workforce Singapore</strong>
    </p>
    """

    # Construct the SendGrid Mail object.
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=html_content
    )

    # Send the email via SendGrid.
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))
