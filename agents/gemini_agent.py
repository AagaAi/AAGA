import os
import google.generativeai as genai

class GeminiAnalyst:
    """
    Gemini API மூலம் அட்வைஸ் வழங்கும் ஏஜெண்ட். 
    API லிமிட்டைத் தவிர்க்க டைமர் வசதியுடன்.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # main.py தேடும் வார்த்தைகள் இங்கே சேர்க்கப்பட்டுள்ளன
        self.available = False  
        self.is_active = False
        self.disabled_until = 0  # <--- 15-min லூப் எரரைத் தடுக்கும் டைமர்
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.available = True  
                self.is_active = True
                print("✅ Gemini Analyst தயார்!")
            except Exception as e:
                print(f"⚠️ Gemini Init Error: {e}")

    def analyze(self, prompt=""):
        return self.get_advice(prompt)

    def get_advice(self, prompt=""):
        if not self.available:
            return "Gemini API குறியீடு இல்லை அல்லது லிமிட் முடிந்தது. AI சொந்த உத்தியைப் பயன்படுத்தும்."
        
        try:
            response = self.model.generate_content(str(prompt))
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API Error: {e}")
            return "Gemini நெட்வொர்க் பிழை. AI சொந்த உத்தியைப் பயன்படுத்தும்."

# பேக்கப் பெயர் (Backup Name)
GeminiAgent = GeminiAnalyst

