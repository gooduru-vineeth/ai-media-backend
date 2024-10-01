import os
import subprocess
import json
from typing import List


class Video:
    def __init__(self, file_path=None):
        self.file_path = None
        if file_path:
            self.load(file_path)

    def load(self, file_path):
        """Load a video file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        self.file_path = file_path

    def trim(self, start_ms, end_ms, output_path, preserve_audio=True):
        """Trim the video from start_ms to end_ms."""
        if not self.file_path:
            raise ValueError(
                "No video loaded. Please load a video file first.")

        duration_ms = self.get_duration()
        if start_ms < 0 or end_ms > duration_ms or start_ms >= end_ms:
            raise ValueError("Invalid start or end time for trimming.")

        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        try:
            cmd = [
                'ffmpeg',
                '-i', self.file_path,
                '-ss', f"{start_sec:.3f}",
                '-to', f"{end_sec:.3f}",
            ]

            if preserve_audio:
                cmd.extend(['-c:a', 'copy'])  # Copy audio without re-encoding
            else:
                cmd.extend(['-an'])  # Remove audio

            cmd.extend(['-c:v', 'libx264', '-preset', 'fast', output_path])

            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error trimming video: {e.stderr.decode()}")

    @staticmethod
    def combine(video_files, output_path, preserve_audio=True):
        """Combine multiple video files."""
        try:
            with open('file_list.txt', 'w') as f:
                for file in video_files:
                    f.write(f"file '{file}'\n")

            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', 'file_list.txt',
            ]

            if preserve_audio:
                cmd.extend(['-c:a', 'aac'])  # Use AAC audio codec
            else:
                cmd.extend(['-an'])  # Remove audio

            cmd.extend(['-c:v', 'libx264', '-preset', 'fast', output_path])

            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error combining videos: {e.stderr.decode()}")
        finally:
            if os.path.exists('file_list.txt'):
                os.remove('file_list.txt')

    def get_duration(self):
        """Get the duration of the video in milliseconds."""
        if not self.file_path:
            raise ValueError(
                "No video loaded. Please load a video file first.")

        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                self.file_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration_seconds = float(data['format']['duration'])
            return int(duration_seconds * 1000)  # Convert to milliseconds
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error getting video duration: {e.stderr}")


def trim_video_api(input_file, output_file, start_ms, end_ms, preserve_audio=True):
    try:
        video = Video(input_file)
        video.trim(start_ms, end_ms, output_file, preserve_audio)
        return {"message": "Video trimmed successfully", "output_file": output_file}
    except Exception as e:
        raise ValueError(f"Error trimming video: {str(e)}")


def combine_videos_api(input_files: List[str], output_file: str, preserve_audio=True):
    try:
        Video.combine(input_files, output_file, preserve_audio)
        return {"message": "Videos combined successfully", "output_file": output_file}
    except Exception as e:
        raise ValueError(f"Error combining videos: {str(e)}")


def get_video_info(file_path):
    """Get video information using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=width,height,duration,bit_rate,codec_name',
            '-of', 'json',
            file_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error getting video info: {e.stderr}")
