import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import wave


class Audio:
    def __init__(self, file_path=None):
        self.audio = None
        if file_path:
            self.load(file_path)

    def load(self, file_path):
        """Load an audio file."""
        _, ext = os.path.splitext(file_path)
        format = ext[1:].lower()  # Remove the dot and convert to lowercase
        try:
            if format == 'wav':
                # Try to open the WAV file to check if it's valid
                with wave.open(file_path, 'rb') as wav_file:
                    params = wav_file.getparams()
                # If it's a valid WAV file, load it with pydub
                self.audio = AudioSegment.from_wav(file_path)
            else:
                self.audio = AudioSegment.from_file(file_path, format=format)
        except (wave.Error, CouldntDecodeError) as e:
            # If WAV-specific or format-specific loading fails, try without specifying the format
            try:
                self.audio = AudioSegment.from_file(file_path)
            except Exception as inner_e:
                raise ValueError(f"Error loading audio file {
                                 file_path}: {str(inner_e)}")

    def trim(self, start_ms, end_ms):
        """Trim the audio from start_ms to end_ms."""
        if not self.audio:
            raise ValueError(
                "No audio loaded. Please load an audio file first.")
        self.audio = self.audio[start_ms:end_ms]

    @staticmethod
    def combine(audio_files):
        """Combine multiple audio files."""
        combined = AudioSegment.empty()
        for file in audio_files:
            try:
                audio = Audio(file)
                combined += audio.audio
            except ValueError as e:
                raise ValueError(f"Error processing file {file}: {str(e)}")
        return combined

    def save(self, output_path, format='mp3'):
        """Save the audio to a file."""
        if not self.audio:
            raise ValueError(
                "No audio to save. Please load or create audio first.")
        self.audio.export(output_path, format=format)

# Example usage in an API context:


def trim_audio_api(input_file, output_file, start_ms, end_ms):
    try:
        audio = Audio(input_file)
        audio.trim(start_ms, end_ms)
        audio.save(output_file)
        return {"message": "Audio trimmed successfully", "output_file": output_file}
    except ValueError as e:
        raise ValueError(f"Error trimming audio: {str(e)}")


def combine_audio_api(input_files, output_file):
    try:
        combined = Audio.combine(input_files)
        combined_audio = Audio()
        combined_audio.audio = combined
        combined_audio.save(output_file)
        return {"message": "Audio files combined successfully", "output_file": output_file}
    except ValueError as e:
        raise ValueError(f"Error combining audio: {str(e)}")
