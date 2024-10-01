from fastapi import APIRouter, HTTPException, Response, UploadFile, File, Form
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from .embedding import router as embedding_router
from .sarvam import SarvamAI, SpeechToTextRequest, TranslateTextRequest, TextToSpeechRequest, SpeechToTextTranslateRequest

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
