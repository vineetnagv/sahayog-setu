import google.generativeai as genai
import pathlib
import os
from dotenv import load_dotenv

load_dotenv('keys.env')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')

def detect_language_and_greet(audio_file_path, context):
    """
    Listens to the first user utterance, identifies the language,
    and returns a formatted string: "LANG_CODE|GREETING_TEXT".
    """
    try:
        # 1. Prepare Audio
        audio_data = {
            'mime_type': 'audio/wav',
            'data': pathlib.Path(audio_file_path).read_bytes()
        }

        # 2. Get User Name
        if context.get('worker'):
            name = context['worker']['name'].split()[0] # "Ramesh"
        else:
            name = "Dost"

        # 3. Prompt for Detection
        prompt = f"""
        You are a language detector for 'Sahayog Setu'.
        Listen to the user's greeting. Identify if they are speaking:
        - Hindi (or Hinglish) -> code: 'hi'
        - Kannada (or Kanglish) -> code: 'kn'
        - Malayalam (or Manglish) -> code: 'ml'
        - English -> code: 'en'

        OUTPUT FORMAT: Return a single string in exactly this format:
        CODE|GREETING

        The GREETING must be a warm welcome using the name '{name}'.
        
        Examples:
        - If Hindi: "hi|Namaste {name}. Sahayog Setu mein swagat hai. Kya aapko kaam chahiye?"
        - If Kannada: "kn|Namaskara {name}. Sahayog Setu ge swagata. Kelasa beka?"
        - If Malayalam: "ml|Namaskaram {name}. Sahayog Setu-ilekku swagatham."
        - If English: "en|Hello {name}. Welcome to Sahayog Setu. Do you need work?"
        
        STRICT RULES:
        1. Use the detected language for the greeting.
        2. Do not add markdown or extra text. Just the CODE|GREETING string.
        """

        response = model.generate_content([prompt, audio_data])
        return response.text.strip()
    except Exception as e:
        print(f"Language Detect Error: {e}")
        # Fallback to Hindi if detection fails
        return "hi|Namaste. Sahayog Setu mein swagat hai."

def get_system_prompt(context, language_code):
    # (Same Logic as before - optimized for the main conversation)
    p_job = context.get('private_job')
    if p_job:
        job_task = p_job['task']
        employer = p_job['employer']
    else:
        job_task = "Kheti ka kaam"
        employer = "Rajesh"

    # Simplified Responses (No Stats in Speech)
    if language_code == 'hi':
        flex_response = (f"Satellite data ke hisaab se aapke gaon mein paani ki kami hai. "
                         f"Isliye maine aapke liye ek accha kaam dhoonda hai. "
                         f"{employer} ke paas '{job_task}' hai. "
                         f"Ismein Chhe sau rupaye milenge.")
        lang_instruction = "Speak Hinglish."
    elif language_code == 'kn':
        flex_response = (f"Satellite data prakara nimma ooralli neeru kammi ide. "
                         f"Adikke naanu nimage uttama kelasa hudukidini. "
                         f"{employer} hathira '{job_task}' ide. "
                         f"Idralli Arunooru rupayi sigutte.")
        lang_instruction = "Speak Kanglish."
    elif language_code == 'ml':
        flex_response = (f"Satellite data anusarichu avide vellam kuravaanu. "
                         f"Athukond njan nalla joli kandu pidichu. "
                         f"{employer}-ude aduthu '{job_task}' undu. "
                         f"Arunooru roopa kittum.")
        lang_instruction = "Speak Manglish."
    else:
        flex_response = (f"Satellite data shows water is low. "
                         f"I found a job with {employer} ('{job_task}'). "
                         f"Pay is Six Hundred rupees.")
        lang_instruction = "Speak English."

    return f"""
    You are 'Sahayog Setu'.
    User Context: Speaking to {context.get('worker', {}).get('name', 'User')}.
    
    RULES:
    1. {lang_instruction}
    2. SCENARIO 1 (User asks for work): Say EXACTLY: "{flex_response}"
    3. SCENARIO 2 (User accepts/says Yes/Ok/Name): Output EXACTLY: "ACCEPT_JOB_PRIVATE"
    4. SCENARIO 3 (User asks for Govt): Output EXACTLY: "ACCEPT_JOB_GOVT"
    """

def get_gemini_response_audio(audio_file_path, context, language_code):
    try:
        audio_data = {'mime_type': 'audio/wav', 'data': pathlib.Path(audio_file_path).read_bytes()}
        generation_config = genai.types.GenerationConfig(temperature=0.15)
        prompt = get_system_prompt(context, language_code)
        response = model.generate_content([prompt, audio_data], generation_config=generation_config)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini AUDIO Error: {e}")
        raise e 

def get_gemini_response_text(user_text, context, language_code):
    try:
        prompt = get_system_prompt(context, language_code)
        full_prompt = f"{prompt}\nUser text: \"{user_text}\""
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini TEXT Error: {e}")
        return "Network error."