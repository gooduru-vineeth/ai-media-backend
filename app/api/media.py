import os
import subprocess
import json
from pydub import AudioSegment
from typing import List, Union, Optional


class Media:
    def __init__(self, file_path: str = None):
        self.file_path = None
        self.media_type = None
        if file_path:
            self.load(file_path)

    def load(self, file_path: str):
        """Load a media file and determine its type."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found: {file_path}")
        self.file_path = file_path
        self.media_type = self._get_media_type()

    def _get_media_type(self) -> str:
        """Determine the media type (audio or video) of the loaded file."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                self.file_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            if 'streams' in data and data['streams']:
                return 'video' if data['streams'][0]['codec_type'] == 'video' else 'audio'
            return 'audio'  # Default to audio if no video stream is found
        except subprocess.CalledProcessError:
            # If ffprobe fails, try to load as audio using pydub
            try:
                AudioSegment.from_file(self.file_path)
                return 'audio'
            except:
                raise ValueError(f"Unable to determine media type for {
                                 self.file_path}")

    def trim(self, start_ms: int, end_ms: int, output_path: str, preserve_audio: bool = True):
        """Trim the media file from start_ms to end_ms."""
        if not self.file_path:
            raise ValueError(
                "No media loaded. Please load a media file first.")

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

            if self.media_type == 'video':
                if preserve_audio:
                    cmd.extend(['-c:a', 'copy'])
                else:
                    cmd.extend(['-an'])
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
            else:  # audio
                cmd.extend(['-acodec', 'copy'])

            cmd.append(output_path)

            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error trimming media: {e.stderr.decode()}")

    @staticmethod
    def combine(media_files: List[str], output_path: str, output_type: str = 'auto', preserve_audio: bool = True):
        """Combine multiple media files."""
        if not media_files:
            raise ValueError("No media files provided for combining.")

        # Determine output type if set to 'auto'
        if output_type == 'auto':
            output_type = Media(media_files[0]).media_type

        try:
            with open('file_list.txt', 'w') as f:
                for file in media_files:
                    f.write(f"file '{file}'\n")

            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', 'file_list.txt',
            ]

            if output_type == 'video':
                if preserve_audio:
                    cmd.extend(['-c:a', 'aac'])
                else:
                    cmd.extend(['-an'])
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
            else:  # audio
                cmd.extend(['-c:a', 'aac'])

            cmd.append(output_path)

            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error combining media: {e.stderr.decode()}")
        finally:
            if os.path.exists('file_list.txt'):
                os.remove('file_list.txt')

    def get_duration(self) -> int:
        """Get the duration of the media in milliseconds."""
        if not self.file_path:
            raise ValueError(
                "No media loaded. Please load a media file first.")

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
            raise ValueError(f"Error getting media duration: {e.stderr}")

    def get_info(self) -> dict:
        """Get media information."""
        if not self.file_path:
            raise ValueError(
                "No media loaded. Please load a media file first.")

        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=width,height,duration,bit_rate,codec_name',
                '-of', 'json',
                self.file_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error getting media info: {e.stderr}")

    def merge_video_audio(self, audio_file: str, output_path: str, keep_original_audio: bool = False,
                          audio_start: Optional[float] = None, audio_end: Optional[float] = None,
                          video_start: Optional[float] = None, video_end: Optional[float] = None,
                          loop_audio: bool = False, fade_audio: bool = False,
                          adjust_video: str = 'none', audio_volume: float = 1.0):
        """
        Merge video with a new audio file.

        :param audio_file: Path to the new audio file
        :param output_path: Path to save the output file
        :param keep_original_audio: Whether to keep the original audio (mix with new audio)
        :param audio_start: Start time of the audio in seconds (optional)
        :param audio_end: End time of the audio in seconds (optional)
        :param video_start: Start time of the video in seconds (optional)
        :param video_end: End time of the video in seconds (optional)
        :param loop_audio: Whether to loop the audio if it's shorter than the video
        :param fade_audio: Whether to apply fade in/out effects to the audio
        :param adjust_video: How to adjust video if audio length differs ('none', 'cut', 'speed')
        :param audio_volume: Volume of the new audio (0.0 to 1.0)
        """
        if self.media_type != 'video':
            raise ValueError(
                "This operation is only applicable to video files.")

        try:
            # Get video and audio durations
            video_duration = self.get_duration() / 1000  # Convert to seconds
            audio_duration = Media(audio_file).get_duration() / 1000

            # Prepare FFmpeg command
            cmd = ['ffmpeg', '-i', self.file_path, '-i', audio_file]

            # Handle audio trimming if specified
            audio_filter = f"volume={audio_volume}"
            if audio_start is not None or audio_end is not None:
                audio_filter += f",atrim={audio_start or 0}:{audio_end or ''}"
            if fade_audio:
                audio_filter += f",afade=t=in:st=0:d=1,afade=t=out:st={
                    audio_end - audio_start - 1 if audio_end else audio_duration - 1}:d=1"

            # Handle video adjustment based on audio length
            video_filter = ""
            if adjust_video == 'cut' and audio_duration < video_duration:
                video_filter = f"trim=0:{audio_duration}"
            elif adjust_video == 'speed':
                speed_factor = audio_duration / video_duration
                video_filter = f"setpts={speed_factor}*PTS"

            # Combine audio and video filters
            filter_complex = f"[1:a]{audio_filter}[a];[0:v]{video_filter}[v]"

            if keep_original_audio:
                filter_complex += ";[0:a][a]amix=inputs=2:duration=longest[outa]"
                map_audio = "[outa]"
            else:
                map_audio = "[a]"

            cmd.extend(['-filter_complex', filter_complex,
                       '-map', '[v]', '-map', map_audio])

            # Handle video trimming if specified
            if video_start is not None:
                cmd.extend(['-ss', str(video_start)])
            if video_end is not None:
                cmd.extend(['-to', str(video_end)])

            # Loop audio if it's shorter than the video and not adjusting video
            if loop_audio and adjust_video == 'none' and audio_duration < video_duration:
                cmd.extend(['-stream_loop', '-1'])

            # Output options
            cmd.extend(['-c:v', 'libx264', '-preset',
                       'fast', '-shortest', output_path])

            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error merging video and audio: {
                             e.stderr.decode()}")

# API functions using the Media class


def trim_media_api(input_file: str, output_file: str, start_ms: int, end_ms: int, preserve_audio: bool = True):
    try:
        media = Media(input_file)
        media.trim(start_ms, end_ms, output_file, preserve_audio)
        return {"message": "Media trimmed successfully", "output_file": output_file}
    except Exception as e:
        raise ValueError(f"Error trimming media: {str(e)}")


def combine_media_api(input_files: List[str], output_file: str, output_type: str = 'auto', preserve_audio: bool = True):
    try:
        # Determine the output type if set to 'auto'
        if output_type == 'auto':
            output_type = Media(input_files[0]).media_type

        # Add the appropriate file extension based on the output type
        if output_type == 'video':
            output_file += '.mp4'
        else:  # audio
            output_file += '.mp3'

        Media.combine(input_files, output_file, output_type, preserve_audio)
        return {"message": "Media combined successfully", "output_file": output_file}
    except Exception as e:
        raise ValueError(f"Error combining media: {str(e)}")


def get_media_info_api(file_path: str):
    try:
        media = Media(file_path)
        return {"media_info": media.get_info()}
    except Exception as e:
        raise ValueError(f"Error getting media info: {str(e)}")

# API function for merging video and audio


def merge_video_audio_api(video_file: str, audio_file: str, output_file: str,
                          keep_original_audio: bool = False, audio_start: Optional[float] = None,
                          audio_end: Optional[float] = None, video_start: Optional[float] = None,
                          video_end: Optional[float] = None, loop_audio: bool = False,
                          fade_audio: bool = False, adjust_video: str = 'none',
                          audio_volume: float = 1.0):
    try:
        media = Media(video_file)
        media.merge_video_audio(audio_file, output_file, keep_original_audio, audio_start,
                                audio_end, video_start, video_end, loop_audio, fade_audio,
                                adjust_video, audio_volume)
        return {"message": "Video and audio merged successfully", "output_file": output_file}
    except Exception as e:
        raise ValueError(f"Error merging video and audio: {str(e)}")
