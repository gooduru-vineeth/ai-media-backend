from fastapi import HTTPException, UploadFile
from pydantic import BaseModel
from typing import List
import os
import logging
from .sarvam import SarvamAI, TextToSpeechRequest
from .asr import transcribe_audio_api_default
from .media import trim_media_api, combine_media_api, merge_video_audio_api
from app.core.embedding import jina_embedding_model
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize SarvamAI
sarvam_ai = SarvamAI(api_key=os.environ['SARVAM_API_KEY'])


class StoryToVideoRequest(BaseModel):
    story: str
    video_descriptions: List[str]


async def create_story_video(request: StoryToVideoRequest):
    try:
        logger.info("Starting story video creation process")

        # Step 1: Text-to-Speech Conversion
        logger.info("Step 1: Converting text to speech")
        tts_request = TextToSpeechRequest(
            inputs=[request.story],
            target_language_code="en-IN",
            speaker="meera",
            pitch=0,
            pace=1,
            loudness=1,
            speech_sample_rate=22050,
            enable_preprocessing=False,
            model="saarika:v1"
        )
        tts_result = await sarvam_ai.text_to_speech(tts_request.inputs, tts_request.target_language_code,
                                                    tts_request.speaker, tts_request.pitch, tts_request.pace,
                                                    tts_request.loudness, tts_request.speech_sample_rate,
                                                    tts_request.enable_preprocessing, tts_request.model)
        audio_file = tts_result['audio_file']
        logger.info(
            f"Text-to-speech conversion completed. Audio file: {audio_file}")

        # Step 2: Automatic Speech Recognition (ASR) for timestamps
        logger.info("Step 2: Performing ASR for timestamps")
        with open(audio_file, "rb") as audio:
            asr_result = await transcribe_audio_api_default(UploadFile(audio))
        logger.info("ASR completed")

        # Step 3: Process video descriptions and fetch relevant videos
        logger.info(
            "Step 3: Processing video descriptions and fetching relevant videos")
        video_segments = []
        for i, description in enumerate(request.video_descriptions):
            logger.info(f"Processing description {
                        i+1}/{len(request.video_descriptions)}")
            # Use Jina embeddings to find relevant videos
            embedding = jina_embedding_model.encode([description])[0].tolist()
            relevant_video = find_relevant_video(embedding)
            logger.info(f"Found relevant video for description {i+1}")

            # Trim the video to match the audio segment
            start_time = asr_result['segments'][i]['start'] * 1000
            end_time = asr_result['segments'][i]['end'] * 1000
            trimmed_video = f"trimmed_video_{i}.mp4"
            trim_media_api(relevant_video, trimmed_video, start_time, end_time)
            video_segments.append(trimmed_video)
            logger.info(f"Trimmed video for segment {i+1}")

        # Step 4: Combine all video segments
        logger.info("Step 4: Combining all video segments")
        combined_video = "combined_video.mp4"
        combine_media_api(video_segments, combined_video, output_type='video')
        logger.info("Video segments combined")

        # Step 5: Merge the combined video with the generated audio
        logger.info("Step 5: Merging combined video with generated audio")
        final_video = "final_story_video.mp4"
        merge_video_audio_api(combined_video, audio_file, final_video)
        logger.info("Video and audio merged")

        # Clean up temporary files
        logger.info("Cleaning up temporary files")
        os.remove(audio_file)
        os.remove(combined_video)
        for video in video_segments:
            os.remove(video)
        logger.info("Temporary files removed")

        logger.info("Story video creation process completed successfully")
        return {"message": "Story video created successfully", "output_file": final_video}

    except Exception as e:
        logger.error(f"Error creating story video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error creating story video: {str(e)}")


def find_relevant_video(embedding):
    # This function should use the embedding to find a relevant video
    # and return the path to that video
    logger.info("Finding relevant video based on embedding")
    # Implement your video search logic here
    # For now, we'll just return a placeholder
    return "placeholder_video.mp4"
