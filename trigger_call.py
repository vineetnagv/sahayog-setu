import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv("keys.env")

# Load credentials from environment variables
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
MY_PHONE = os.getenv("MY_PHONE")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# This URL tells Twilio what to do when you answer the phone.
# It points to the SAME endpoint you already built!
NGROK_URL = "https://prebroadcasting-kaitlynn-noncartelized.ngrok-free.dev" 
WEBHOOK_URL = f"{NGROK_URL}/voice/incoming"

print(f"Calling {MY_PHONE}...")

call = client.calls.create(
    to=MY_PHONE,
    from_=TWILIO_NUMBER,
    url=WEBHOOK_URL,
    method="POST"
)

print(f"Call initiated! SID: {call.sid}")