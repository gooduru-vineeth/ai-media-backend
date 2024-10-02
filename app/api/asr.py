import whisper
import os
import torch


class AutomaticSpeechRecognition:
    _instance = None

    def __init__(self, model_name="tiny"):
        try:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = whisper.load_model(model_name).to(self.device)
            self.model_name = model_name
        except RuntimeError:
            print(f"Error using MPS. Falling back to CPU for {
                  model_name} model.")
            self.device = "cpu"
            self.model = whisper.load_model(model_name)
            self.model_name = model_name

    @classmethod
    def get_instance(cls, model_name="turbo"):
        if not cls._instance or cls._instance.model_name != model_name:
            print(f"Creating new instance with {model_name} model.")
            cls._instance = cls(model_name)
        return cls._instance

    def transcribe(self, audio_file_path):
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            result = self.model.transcribe(
                audio_file_path, word_timestamps=True, fp16=False)
        except RuntimeError:
            print(f"Error using {
                  self.device} for transcription. Falling back to CPU.")
            self.device = "cpu"
            self.model = self.model.to(self.device)
            result = self.model.transcribe(
                audio_file_path, word_timestamps=True, fp16=False)

        return {
            "text": result["text"],
            "words": self._extract_words(result)
        }

    def _extract_words(self, result):
        words = []
        for segment in result["segments"]:
            for word_info in segment["words"]:
                words.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"]
                })
        return words

# API functions


def transcribe_audio_api(audio_file_path: str, model_name: str = "tiny"):
    try:
        asr = AutomaticSpeechRecognition(model_name)
        result = asr.transcribe(audio_file_path)
        return result
    except Exception as e:
        raise ValueError(f"Error transcribing audio: {str(e)}")


def transcribe_audio_api_default(audio_file_path: str):
    try:
        # Use the singleton instance with "turbo" model
        asr = AutomaticSpeechRecognition.get_instance("turbo")
        result = asr.transcribe(audio_file_path)
        return result
    except Exception as e:
        raise ValueError(f"Error transcribing audio: {str(e)}")
