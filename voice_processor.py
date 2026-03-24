"""
Voice Processing: Whisper for STT, pyttsx3 for TTS
"""
import os
import tempfile
import speech_recognition as sr
import pyttsx3
from typing import Optional

class VoiceProcessor:
    def __init__(self, model_size: str = "tiny"):
        print(f"Loading Whisper model: {model_size}...")
        
        # Import whisper here to avoid errors if not installed
        try:
            import whisper # type: ignore
            self.whisper_model = whisper.load_model(model_size)
        except ImportError:
            print("⚠️  Whisper not available, using speech recognition only")
            self.whisper_model = None
        
        self.recognizer = sr.Recognizer()
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 175)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
        except Exception as e:
            print(f"⚠️  TTS init error: {e}")
            self.tts_engine = None
    
    def record_audio(self, duration: Optional[int] = None) -> str:
        """Record audio from microphone"""
        try:
            with sr.Microphone() as source:
                print("🎙️ Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                if duration:
                    audio = self.recognizer.record(source, duration=duration)
                else:
                    audio = self.recognizer.listen(source, timeout=5)
                
                # Use Whisper if available, otherwise use Google Speech Recognition
                if self.whisper_model:
                    print("Processing with Whisper...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio.get_wav_data())
                        tmp_path = tmp.name
                    
                    try:
                        result = self.whisper_model.transcribe(tmp_path)
                        text = result["text"].strip()
                    finally:
                        os.unlink(tmp_path)
                else:
                    print("Processing with Google...")
                    text = self.recognizer.recognize_google(audio)
                
                print(f"Heard: '{text}'")
                return text
                
        except sr.WaitTimeoutError:
            return "No speech detected"
        except Exception as e:
            print(f"Audio error: {e}")
            raise e
    
    def speak(self, text: str):
        """Text-to-speech"""
        if self.tts_engine is None:
            print(f"🔊 Would speak: {text[:50]}...")
            return
            
        try:
            print(f"🔊 Speaking...")
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")