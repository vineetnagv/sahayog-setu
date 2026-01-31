import speech_recognition as sr

def transcribe_with_google(audio_file_path, lang_code):
    """
    Fallback ASR using Google's Standard Speech API.
    Supports EN, HI, KN, ML.
    """
    recognizer = sr.Recognizer()
    
    # Map our internal codes (en, hi, kn, ml) to Google's BCP-47 codes
    google_lang_map = {
        'en': 'en-IN',
        'hi': 'hi-IN',
        'kn': 'kn-IN',
        'ml': 'ml-IN'
    }
    
    target_lang = google_lang_map.get(lang_code, 'hi-IN')
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            # Record the audio from the file
            audio_data = recognizer.record(source)
            
            # Send to Google Speech API
            # Note: This uses the default free key. For production, you'd need a real key,
            # but for hackathons, this works perfectly.
            text = recognizer.recognize_google(audio_data, language=target_lang)
            return text
            
    except sr.UnknownValueError:
        # Google couldn't understand the audio
        return ""
    except sr.RequestError:
        # API was unreachable
        return ""
    except Exception as e:
        print(f"Fallback ASR Error: {e}")
        return ""