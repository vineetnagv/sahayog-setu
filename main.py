from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
import requests
import os
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv

load_dotenv('keys.env')

# Import Services
from sms_service import send_job_sms
from universal_asr import transcribe_with_google
from ai_service import get_gemini_response_audio, get_gemini_response_text, detect_language_and_greet
from db_service import get_call_context

app = FastAPI()
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# --- 1. START: ASK USER TO SPEAK (NO BUTTONS) ---
@app.post("/voice/incoming")
async def incoming_call(request: Request):
    response = VoiceResponse()
    
    # Simple open-ended prompt
    # We say "Namaste" to invite them to speak
    response.say("Namaste. Sahayog Setu.", language="hi-IN")
    
    # Record the user saying "Hello" or "Namaste" or "Namaskara"
    # Action goes to the new identification route
    response.record(action='/voice/identify-language', max_length=3, play_beep=True)
    
    return Response(content=str(response), media_type="application/xml")

# --- 2. IDENTIFY LANGUAGE & GREET ---
@app.post("/voice/identify-language")
async def identify_language(RecordingUrl: str = Form(...), CallSid: str = Form(...), From: str = Form(...)):
    print(f"🎧 Identifying Language for Caller: {From}...")
    
    # 1. Get Context (Name)
    call_context = get_call_context(From)
    
    # 2. Download Audio
    final_download_url = f"{RecordingUrl}.wav"
    audio_filename = f"intro_{CallSid}.wav"
    
    try:
        audio_data = requests.get(final_download_url, auth=(ACCOUNT_SID, AUTH_TOKEN))
        with open(audio_filename, 'wb') as f:
            f.write(audio_data.content)
        
        # 3. AI Detection
        # Returns format: "hi|Namaste Ramesh..."
        result_string = detect_language_and_greet(audio_filename, call_context)
        print(f"🤖 Detection Result: {result_string}")
        
        # 4. Parse Result
        if "|" in result_string:
            detected_lang, greeting_text = result_string.split("|", 1)
        else:
            # Safety Fallback
            detected_lang = "hi"
            greeting_text = "Namaste. Sahayog Setu mein swagat hai."
        
        # Clean up
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        # 5. Respond & Switch to Main Loop
        response = VoiceResponse()
        response.say(greeting_text, language="hi-IN") # Twilio's Hindi engine reads English chars reasonably well
        
        # Pass the detected language to the next step via query param
        response.record(action=f'/voice/process-recording?lang={detected_lang}', max_length=5, play_beep=True)
        return Response(content=str(response), media_type="application/xml")

    except Exception as e:
        print(f"Error in ID: {e}")
        response = VoiceResponse()
        response.say("Network error. Please call again.")
        return Response(content=str(response), media_type="application/xml")

# --- 3. MAIN CONVERSATION LOOP (Standard) ---
@app.post("/voice/process-recording")
async def process_recording(lang: str, RecordingUrl: str = Form(...), CallSid: str = Form(...), From: str = Form(...)):
    print(f"Processing Call From: {From} in Language: {lang}")
    
    call_context = get_call_context(From)
    final_download_url = f"{RecordingUrl}.wav"
    audio_filename = f"rec_{CallSid}.wav"

    try:
        audio_data = requests.get(final_download_url, auth=(ACCOUNT_SID, AUTH_TOKEN))
        with open(audio_filename, 'wb') as f:
            f.write(audio_data.content)
            
        ai_reply = ""
        try:
            print(f"Sending Audio + Context to Gemini...")
            ai_reply = get_gemini_response_audio(audio_filename, call_context, lang)
            print("Gemini Response:", ai_reply)
        except Exception as e:
            print(f"FALLBACK: {e}")
            transcribed_text = transcribe_with_google(audio_filename, lang)
            if transcribed_text:
                ai_reply = get_gemini_response_text(transcribed_text, call_context, lang)
            else:
                ai_reply = "Network weak. Please try again."

        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        response = VoiceResponse()

        if "ACCEPT_JOB" in ai_reply:
            if "PRIVATE" in ai_reply: job_taken = call_context.get('private_job')
            elif "GOVT" in ai_reply: job_taken = call_context.get('govt_job')
            else: job_taken = call_context.get('private_job')

            confirms = {
                'en': "Job confirmed. Check SMS.",
                'hi': "Kaam pakka ho gaya. SMS check karein.",
                'kn': "Kelasa confirm aagide. SMS nodi.",
                'ml': "Joli urappayi. SMS parishodhikku."
            }
            response.say(confirms.get(lang, confirms['hi']), language="hi-IN")
            
            if job_taken:
                try:
                    send_job_sms(From, job_taken)
                    print(f"SMS Sent")
                except: pass
            
            response.hangup()
        else:
            response.say(ai_reply, language="hi-IN")
            response.record(action=f'/voice/process-recording?lang={lang}', max_length=5, play_beep=True)
            
        return Response(content=str(response), media_type="application/xml")

    except Exception as e:
        print(f"Server Error: {e}")
        return Response(content=str(VoiceResponse().hangup()), media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)