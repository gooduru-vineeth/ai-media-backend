import os
from google.cloud import texttospeech
from google.auth.exceptions import DefaultCredentialsError


class GoogleTTS:
    def __init__(self):
        self.client = None
        try:
            self.client = texttospeech.TextToSpeechClient()
        except DefaultCredentialsError:
            print("Error: Google Cloud credentials not found.")
            print("Please set up your credentials by following these steps:")
            print("1. Run 'gcloud auth application-default login' in your terminal.")
            print("2. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable:")
            print(
                "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/credentials.json'")

    def text_to_speech(self, text, language_code="en-US", voice_name=None,
                       ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
                       audio_encoding=texttospeech.AudioEncoding.MP3):
        if not self.client:
            raise Exception(
                "Google TTS client not initialized. Check your credentials.")

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=ssml_gender
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding
        )

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        return response.audio_content


# Initialize the GoogleTTS class
google_tts = GoogleTTS()
