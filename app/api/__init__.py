from fastapi import APIRouter, HTTPException, Response, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
from .embedding import router as embedding_router
from .sarvam import SarvamAI, SpeechToTextRequest, TranslateTextRequest, TextToSpeechRequest, SpeechToTextTranslateRequest
from .audio import Audio, trim_audio_api, combine_audio_api
from .video import Video, trim_video_api, combine_videos_api, get_video_info
from .media import trim_media_api, combine_media_api, get_media_info_api, merge_video_audio_api

load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

router = APIRouter()
router.include_router(embedding_router, tags=["embedding"])

# Initialize SarvamAI
sarvam_ai = SarvamAI(api_key=os.environ['SARVAM_API_KEY'])


class StoryPrompt(BaseModel):
    prompt: str


@router.post("/generate-story")
async def generate_story(story_prompt: StoryPrompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")

        response = model.generate_content(
            f"Generate a short story based on the following prompt: {
                story_prompt.prompt}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
        )

        return {"story": response.text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating story: {str(e)}")


@router.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    language_code: str = Form(...),
    model: str = Form("saarika:v1")
):
    try:
        request = SpeechToTextRequest(language_code=language_code, model=model)
        result = sarvam_ai.speech_to_text(
            file, request.language_code, request.model)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in speech to text: {str(e)}")


@router.post("/translate-text")
async def translate_text(request: TranslateTextRequest):
    try:
        result = sarvam_ai.translate_text(
            request.input, request.source_language_code, request.target_language_code,
            request.speaker_gender, request.mode, request.model, request.enable_preprocessing
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in text translation: {str(e)}")


@router.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        result = sarvam_ai.text_to_speech(
            request.inputs, request.target_language_code, request.speaker,
            request.pitch, request.pace, request.loudness, request.speech_sample_rate,
            request.enable_preprocessing, request.model
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in text to speech: {str(e)}")


@router.post("/speech-to-text-translate")
async def speech_to_text_translate(
    file: UploadFile = File(...),
    prompt: str = Form(None),
    model: str = Form("saaras:v1")
):
    try:
        request = SpeechToTextTranslateRequest(prompt=prompt, model=model)
        result = sarvam_ai.speech_to_text_translate(
            file, request.prompt, request.model)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in speech to text translation: {str(e)}")


class TrimAudioRequest(BaseModel):
    start_ms: int
    end_ms: int


@router.post("/trim-audio")
async def trim_audio(
    file: UploadFile = File(...),
    start_ms: int = Form(...),
    end_ms: int = Form(...)
):
    try:
        trim_request = TrimAudioRequest(start_ms=start_ms, end_ms=end_ms)
        # Create a temporary file to store the uploaded audio
        temp_input = f"temp_input_{file.filename}"
        temp_output = f"temp_output_{file.filename}"

        with open(temp_input, "wb") as buffer:
            buffer.write(await file.read())

        result = trim_audio_api(temp_input, temp_output,
                                trim_request.start_ms, trim_request.end_ms)

        # Read the trimmed audio file
        with open(temp_output, "rb") as trimmed_file:
            trimmed_audio_data = trimmed_file.read()

        # Clean up temporary files
        os.remove(temp_input)
        os.remove(temp_output)

        # Return the trimmed audio as a response
        return Response(
            content=trimmed_audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=trimmed_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error trimming audio: {str(e)}")


@router.post("/combine-audio")
async def combine_audio(files: List[UploadFile] = File(...)):
    temp_files = []
    temp_output = "temp_combined_output.mp3"
    try:
        for file in files:
            temp_file = f"temp_{file.filename}"
            with open(temp_file, "wb") as buffer:
                buffer.write(await file.read())
            temp_files.append(temp_file)

        result = combine_audio_api(temp_files, temp_output)

        # Read the combined audio file
        with open(temp_output, "rb") as combined_file:
            combined_audio_data = combined_file.read()

        # Return the combined audio as a response
        return Response(
            content=combined_audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=combined_audio.mp3"}
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error combining audio: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_output):
            os.remove(temp_output)


@router.post("/trim-video")
async def trim_video(
    file: UploadFile = File(...),
    start_ms: int = Form(...),
    end_ms: int = Form(...),
    preserve_audio: bool = Form(True)
):
    try:
        temp_input = f"temp_input_{file.filename}"
        temp_output = f"temp_output_{file.filename}"

        with open(temp_input, "wb") as buffer:
            buffer.write(await file.read())

        result = trim_video_api(temp_input, temp_output,
                                start_ms, end_ms, preserve_audio)

        # Read the trimmed video file
        with open(temp_output, "rb") as trimmed_file:
            trimmed_video_data = trimmed_file.read()

        # Clean up temporary files
        os.remove(temp_input)
        os.remove(temp_output)

        # Return the trimmed video as a response
        return Response(
            content=trimmed_video_data,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=trimmed_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error trimming video: {str(e)}")


@router.post("/combine-videos")
async def combine_videos(
    files: List[UploadFile] = File(...),
    preserve_audio: bool = Form(True)
):
    temp_files = []
    temp_output = "temp_combined_output.mp4"
    try:
        for file in files:
            temp_file = f"temp_{file.filename}"
            with open(temp_file, "wb") as buffer:
                buffer.write(await file.read())
            temp_files.append(temp_file)

        result = combine_videos_api(temp_files, temp_output, preserve_audio)

        # Read the combined video file
        with open(temp_output, "rb") as combined_file:
            combined_video_data = combined_file.read()

        # Return the combined video as a response
        return Response(
            content=combined_video_data,
            media_type="video/mp4",
            headers={
                "Content-Disposition": "attachment; filename=combined_video.mp4"}
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error combining videos: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_output):
            os.remove(temp_output)


@router.post("/video-info")
async def video_info(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())

        info = get_video_info(temp_file)
        return {"video_info": info}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting video info: {str(e)}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@router.post("/trim-media")
async def trim_media(
    file: UploadFile = File(...),
    start_ms: int = Form(...),
    end_ms: int = Form(...),
    preserve_audio: bool = Form(True)
):
    try:
        temp_input = f"temp_input_{file.filename}"
        temp_output = f"temp_output_{file.filename}"

        with open(temp_input, "wb") as buffer:
            buffer.write(await file.read())

        result = trim_media_api(temp_input, temp_output,
                                start_ms, end_ms, preserve_audio)

        # Read the trimmed media file
        with open(temp_output, "rb") as trimmed_file:
            trimmed_media_data = trimmed_file.read()

        # Clean up temporary files
        os.remove(temp_input)
        os.remove(temp_output)

        # Return the trimmed media as a response
        return Response(
            content=trimmed_media_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=trimmed_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error trimming media: {str(e)}")


@router.post("/combine-media")
async def combine_media(
    files: List[UploadFile] = File(...),
    output_type: str = Form("auto"),
    preserve_audio: bool = Form(True)
):
    temp_files = []
    temp_output = "temp_combined_output"
    try:
        for file in files:
            temp_file = f"temp_{file.filename}"
            with open(temp_file, "wb") as buffer:
                buffer.write(await file.read())
            temp_files.append(temp_file)

        result = combine_media_api(
            temp_files, temp_output, output_type, preserve_audio)

        # The output file now has the correct extension
        output_file = result["output_file"]

        # Read the combined media file
        with open(output_file, "rb") as combined_file:
            combined_media_data = combined_file.read()

        # Determine the appropriate media type
        media_type = "video/mp4" if output_file.endswith(
            ".mp4") else "audio/mpeg"

        # Return the combined media as a response
        return Response(
            content=combined_media_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_file)}"}
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error combining media: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(output_file):
            os.remove(output_file)


@router.post("/media-info")
async def media_info(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())

        info = get_media_info_api(temp_file)
        return {"media_info": info}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting media info: {str(e)}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@router.post("/merge-video-audio")
async def merge_video_audio(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    keep_original_audio: bool = Form(False),
    audio_start: Optional[float] = Form(None),
    audio_end: Optional[float] = Form(None),
    video_start: Optional[float] = Form(None),
    video_end: Optional[float] = Form(None),
    loop_audio: bool = Form(False),
    fade_audio: bool = Form(False)
):
    try:
        temp_video = f"temp_video_{video_file.filename}"
        temp_audio = f"temp_audio_{audio_file.filename}"
        temp_output = f"temp_output_merged.mp4"

        # Save uploaded files
        with open(temp_video, "wb") as buffer:
            buffer.write(await video_file.read())
        with open(temp_audio, "wb") as buffer:
            buffer.write(await audio_file.read())

        result = merge_video_audio_api(temp_video, temp_audio, temp_output,
                                       keep_original_audio, audio_start, audio_end,
                                       video_start, video_end, loop_audio, fade_audio)

        # Read the merged file
        with open(temp_output, "rb") as merged_file:
            merged_data = merged_file.read()

        # Clean up temporary files
        os.remove(temp_video)
        os.remove(temp_audio)
        os.remove(temp_output)

        # Return the merged video as a response
        return Response(
            content=merged_data,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=merged_video_audio.mp4"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error merging video and audio: {str(e)}")
