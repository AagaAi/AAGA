import os
import google.generativeai as genai

class GeminiAnalyst:
    """
    Gemini API மூலம் அட்வைஸ் வழங்கும் ஏஜெண்ட். 
    API வேலை செய்யவில்லை என்றாலும் எரர் அடிக்காமல் சிஸ்டத்தைக் காப்பாற்றும்.
    """
    # பிழை சரி செய்யப்பட்டது: main.py அனுப்பும் api_key-ஐ உள்ளே வாங்கிக் கொள்ளும்
    def __init__(self, api_key=None):
        # main.py அனுப்பும் key அல்லது Environment-ல் உள்ள key என இரண்டில் ஒன்றை எடுத்துக்கொள்ளும்
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.is_active = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # லேட்டஸ்ட் மாடலான gemini-1.5-flash-ஐப் பயன்படுத்துகிறோம்
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.is_active = True
                print("✅ Gemini Analyst தயார்!")
            except Exception as e:
                print(f"⚠️ Gemini Init Error: {e}")

    def analyze(self, prompt=""):
        return self.get_advice(prompt)

    def get_advice(self, prompt=""):
        """Gemini-யிடம் இருந்து பதிலைப் பெறுகிறது"""
        if not self.is_active:
            return "Gemini API குறியீடு இல்லை (அல்லது) லிமிட் முடிந்தது. AI தனது சொந்த உத்திகளைப் பயன்படுத்தி ட்ரேட் செய்யும்."
        
        try:
            # ஸ்ட்ரிங் பார்மட்டிற்கு மாற்றி அனுப்புகிறோம்
            response = self.model.generate_content(str(prompt))
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API Error: {e}")
            return "Gemini நெட்வொர்க் பிழை. AI சொந்த உத்தியைப் பயன்படுத்துகிறது."

# ஒருவேளை வேறு ஏதேனும் கோப்பு பழைய பெயரில் தேடினால் எரர் வராமல் இருக்க:
GeminiAgent = GeminiAnalyst

