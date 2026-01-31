import google.generativeai as genai

# PASTE YOUR API KEY HERE
GOOGLE_API_KEY = "AIzaSyB-t8Q17z150l7uuK_j-rljvfTOJtYKl3U"

genai.configure(api_key=GOOGLE_API_KEY)

print("Checking available models for your API key...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"FOUND: {m.name}")
except Exception as e:
    print(f"Error: {e}")