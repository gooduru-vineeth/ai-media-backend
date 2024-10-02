import requests
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class SarvamAI:
    def __init__(self, api_key: str, base_url: str = "https://api.sarvam.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "api-subscription-key": self.api_key,
        }

    def speech_to_text(self, file: UploadFile, language_code: str, model: str = "saarika:v1"):
        endpoint = f"{self.base_url}/speech-to-text"
        files = {"file": (file.filename, file.file, file.content_type)}
        data = {
            "language_code": language_code,
            "model": model
        }
        response = requests.post(
            endpoint, headers=self.headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.text)

    def translate_text(self, input_text: str, source_language_code: str, target_language_code: str,
                       speaker_gender: str = "Female", mode: str = "formal",
                       model: str = "mayura:v1", enable_preprocessing: bool = False):
        endpoint = f"{self.base_url}/translate"
        data = {
            "input": input_text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "speaker_gender": speaker_gender,
            "mode": mode,
            "model": model,
            "enable_preprocessing": enable_preprocessing
        }
        response = requests.post(endpoint, headers=self.headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.text)

    def text_to_speech(self, inputs: list[str], target_language_code: str, speaker: str = "meera",
                       pitch: float = None, pace: float = None, loudness: float = None,
                       speech_sample_rate: int = 22050, enable_preprocessing: bool = False,
                       model: str = "bulbul:v1"):
        endpoint = f"{self.base_url}/text-to-speech"
        data = {
            "inputs": inputs,
            "target_language_code": target_language_code,
            "speaker": speaker,
            "pitch": pitch,
            "pace": pace,
            "loudness": loudness,
            "speech_sample_rate": speech_sample_rate,
            "enable_preprocessing": enable_preprocessing,
            "model": model
        }
        response = requests.post(endpoint, headers=self.headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.text)

    def speech_to_text_translate(self, file: UploadFile, prompt: str = None, model: str = "saaras:v1"):
        endpoint = f"{self.base_url}/speech-to-text-translate"
        files = {"file": (file.filename, file.file, file.content_type)}
        data = {
            "prompt": prompt,
            "model": model
        }
        response = requests.post(
            endpoint, headers=self.headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.text)

# Pydantic models for request bodies with validations


class SpeechToTextRequest(BaseModel):
    language_code: str = Field(
        "hi-IN", description="Model to be used for speech to text")
    model: str = Field(
        "saarika:v1", description="Model to be used for speech to text")

    @field_validator('language_code')
    @classmethod
    def validate_language_code(cls, v):
        valid_codes = ['hi-IN', 'bn-IN', 'kn-IN', 'ml-IN',
                       'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'gu-IN']
        if v not in valid_codes:
            raise ValueError(
                f"Invalid language_code. Must be one of {valid_codes}")
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v != "saarika:v1":
            raise ValueError(
                "Invalid model. Currently only 'saarika:v1' is supported")
        return v


class TranslateTextRequest(BaseModel):
    input: str = Field(
        "Hello , how are you ?", description="Input text to be translated")
    source_language_code: str = Field(
        "en-IN", description="Source language code")
    target_language_code: str = Field(
        "hi-IN", description="Target language code")
    speaker_gender: str = Field(
        "Female", description="Gender of the speaker")
    mode: str = Field("formal", description="Translation mode")
    model: str = Field(
        "mayura:v1", description="Model to be used for translation")
    enable_preprocessing: bool = Field(
        False, description="Enable custom preprocessing")

    @field_validator('source_language_code')
    @classmethod
    def validate_source_language(cls, v):
        if v != "en-IN":
            raise ValueError(
                "Invalid source_language_code. Currently only 'en-IN' is supported")
        return v

    @field_validator('target_language_code')
    @classmethod
    def validate_target_language(cls, v):
        valid_codes = ['hi-IN', 'bn-IN', 'kn-IN', 'ml-IN',
                       'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'gu-IN']
        if v not in valid_codes:
            raise ValueError(
                f"Invalid target_language_code. Must be one of {valid_codes}")
        return v

    @field_validator('speaker_gender')
    @classmethod
    def validate_speaker_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError(
                "Invalid speaker_gender. Must be either 'Male' or 'Female'")
        return v

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v not in ["formal", "code-mixed"]:
            raise ValueError(
                "Invalid mode. Must be either 'formal' or 'code-mixed'")
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v != "mayura:v1":
            raise ValueError(
                "Invalid model. Currently only 'mayura:v1' is supported")
        return v


class TextToSpeechRequest(BaseModel):
    inputs: List[str] = Field(
        ["Hello , how are you ?"], description="List of input sentences to be converted to speech")
    target_language_code: str = Field(
        "en-IN", description="Language code")
    speaker: str = Field(
        "meera", description="Speaker for the generated speech")
    pitch: Optional[float] = Field(
        None, description="Pitch of the generated speech")
    pace: Optional[float] = Field(
        None, description="Pace or speed of the generated speech")
    loudness: Optional[float] = Field(
        None, description="Volume of the generated speech")
    speech_sample_rate: int = Field(
        22050, description="Sample rate of the generated speech, higher the better quality, Available options: 8000, 16000, 22050")
    enable_preprocessing: bool = Field(
        False, description="Enable custom preprocessing")
    model: str = Field(
        "bulbul:v1", description="Model to be used for text to speech")

    @field_validator('target_language_code')
    @classmethod
    def validate_target_language(cls, v):
        valid_codes = ['hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN',
                       'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'en-IN', 'gu-IN']
        if v not in valid_codes:
            raise ValueError(
                f"Invalid target_language_code. Must be one of {valid_codes}")
        return v

    @field_validator('speaker')
    @classmethod
    def validate_speaker(cls, v):
        valid_speakers = ['meera', 'pavithra',
                          'maitreyi', 'arvind', 'amol', 'amartya']
        if v not in valid_speakers:
            raise ValueError(f"Invalid speaker. Must be one of {
                             valid_speakers}")
        return v

    @field_validator('speech_sample_rate')
    @classmethod
    def validate_speech_sample_rate(cls, v):
        valid_rates = [8000, 16000, 22050]
        if v not in valid_rates:
            raise ValueError(
                f"Invalid speech_sample_rate. Must be one of {valid_rates}")
        return v

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v != "bulbul:v1":
            raise ValueError(
                "Invalid model. Currently only 'bulbul:v1' is supported")
        return v


class SpeechToTextTranslateRequest(BaseModel):
    prompt: Optional[str] = Field(
        None, description="Prompt to assist the transcription")
    model: str = Field(
        "saaras:v1", description="Model to be used for speech to text translation")

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v != "saaras:v1":
            raise ValueError(
                "Invalid model. Currently only 'saaras:v1' is supported")
        return v
