import os
from twilio.rest import Client
from dotenv import load_dotenv

# --- FIX: Load 'keys.env' explicitly ---
load_dotenv('keys.env')

def send_job_sms(to_number, job_details):
    """
    Sends a confirmation SMS to the worker.
    Dynamically formats the message based on job type (Govt vs Private).
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    
    # Safety Check
    if not account_sid or not auth_token:
        print("❌ SMS ERROR: Twilio keys missing in keys.env")
        return

    try:
        client = Client(account_sid, auth_token)
        
        # Format the message based on Job Type
        # If it's a Government Job (NREGA), formatting is slightly different
        if job_details.get('type') and "Government" in job_details['type']:
             body_text = (f"Sahayog Setu: Job Confirmed!\n"
                         f"Type: {job_details['task']} (NREGA)\n"
                         f"Wage: Rs. {job_details['wage']}/day\n"
                         f"Report to: {job_details['location']}\n"
                         f"Employer: Panchayat")
        else:
             # Standard Private/Farm Job
             body_text = (f"Sahayog Setu: Job Confirmed!\n"
                         f"Type: {job_details['task']}\n"
                         f"Wage: Rs. {job_details['wage']}/day\n"
                         f"Location: {job_details['location']}\n"
                         f"Farmer: {job_details['farmer_name']}")

        # Send the message
        # Note: Ensure 'from_' is your actual Twilio phone number
        message = client.messages.create(
            body=body_text,
            from_='+18154918027', # Replace with your Twilio Number if different
            to=to_number
        )
        print(f"✅ SMS Sent! SID: {message.sid}")
        
    except Exception as e:
        print(f"❌ SMS FAILED: {e}")