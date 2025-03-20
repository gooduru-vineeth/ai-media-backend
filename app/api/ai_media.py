import concurrent.futures
import subprocess
from database import MongoDB
from pymilvus import connections
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel
from typing import List
import os
import logging
from google.cloud import texttospeech
from .asr import transcribe_audio_api_default
from .media import trim_media_api, merge_video_audio_api, Media
from app.core.embedding import jina_embedding_model
from dotenv import load_dotenv
import aiofiles
import google.generativeai as genai
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import MilvusClient
load_dotenv()

# Initialize MongoDB connection
mongo_db = MongoDB(
    os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
    os.getenv("MONGODB_DB_NAME", "image_analysis_db")
)
conn = MilvusClient()
collection_name = "image_analysis_2"


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Google Cloud Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient()

# Initialize Gemini API
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))


class StoryToVideoRequest(BaseModel):
    story: str
    video_descriptions: List[str]


async def create_story_video(request: StoryToVideoRequest):
    try:
        # logger.info("Starting story video creation process")

        # # Step 1: Text-to-Speech Conversion using Google Cloud
        # logger.info("Step 1: Converting text to speech using Google Cloud TTS")
        # audio_file = "story_audio.wav"
        # text_to_speech(request.story, audio_file)
        # logger.info(
        #     f"Text-to-speech conversion completed. Audio file: {audio_file}")

        # # Step 2: Automatic Speech Recognition (ASR) for timestamps
        # logger.info("Step 2: Performing ASR for timestamps")
        # async with aiofiles.open(audio_file, "rb") as audio:
        #     audio_content = await audio.read()

        # # Create a temporary file to store the audio content
        # temp_audio_file = "temp_story_audio.wav"
        # async with aiofiles.open(temp_audio_file, "wb") as temp_file:
        #     await temp_file.write(audio_content)

        # # Use UploadFile with the temporary file
        # audio_upload = UploadFile(
        #     filename="story_audio.wav", file=open(temp_audio_file, "rb"))
        # asr_result = await transcribe_audio_api_default(audio_upload)
        # # Close and remove the temporary file
        # audio_upload.file.close()
        # os.remove(temp_audio_file)

        # logger.info("ASR completed")
        # audio_duration = asr_result.get('words')[-1].get('end')
        # print(audio_duration)
        # # Step 3: Process video descriptions and fetch relevant videos
        # logger.info(
        #     "Step 3: Processing video descriptions and fetching relevant videos")
        # video_descriptions = await generate_video_descriptions(
        #     request.story, audio_duration)
        data = await generate_video_descriptions_test(
            "", 102)
        video_descriptions = data.get('data')
        video_segments = []
        i = 0
        for video_details in video_descriptions:
            i += 1
            description = video_details.get('description')
            start_time = video_details.get('start_time')
            end_time = video_details.get('end_time')
            # print(video_details)

            logger.info(f"Processing description {
                        i+1}/{len(video_descriptions)}")

            relevant_videos = await find_relevant_videos(description, 5, end_time - start_time)
            for video in relevant_videos:
                video_segments.append(video)
        #  combine videos
        combined_video = combine_media_api(
            input_files=video_segments,
            output_file="combined_video.mp4",
            output_type='video'
        )
        print("Debug: relevant_videos =", relevant_videos)  # Add this line
        logger.info(f"Found relevant video for description {i+1}")

        # # Step 4: Combine all video segments
        # logger.info("Step 4: Combining all video segments")
        # combined_video = "combined_video.mp4"
        # combine_media_api(video_segments, combined_video, output_type='video')
        # logger.info("Video segments combined")

        # # Step 5: Merge the combined video with the generated audio
        # logger.info("Step 5: Merging combined video with generated audio")
        # final_video = "final_story_video.mp4"
        # merge_video_audio_api(combined_video, audio_file, final_video)
        # logger.info("Video and audio merged")

        # # Clean up temporary files
        # logger.info("Cleaning up temporary files")
        # os.remove(audio_file)
        # os.remove(combined_video)
        # for video in video_segments:
        #     os.remove(video)
        # logger.info("Temporary files removed")

        # logger.info("Story video creation process completed successfully")
        # return {"message": "Story video created successfully", "output_file": final_video}

    except Exception as e:
        logger.error(f"Error creating story video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error creating story video: {str(e)}")


def text_to_speech(text, output_file):
    try:
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Journey-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding="LINEAR16",
            effects_profile_id=[
                "small-bluetooth-speaker-class-device"
            ],
            pitch=0,
            speaking_rate=0
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        # The response's audio_content is binary
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        logger.info(f'Audio content written to file "{output_file}"')

    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        raise

# # Single vector search
# res = client.search(
#     collection_name="test_collection", # Replace with the actual name of your collection
#     # Replace with your query vector
#     data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
#     limit=5, # Max. number of search results to return
#     search_params={"metric_type": "IP", "params": {}} # Search parameters
# )

# # Convert the output to a formatted JSON string
# result = json.dumps(res, indent=4)
# print(result)


async def search_similar_documents(query_text, limit=5):
    # Generate embedding for the query text using Jina embedding model
    query_embedding = jina_embedding_model.encode([query_text])[0].tolist()

    # Load the collection into memory for searching

    # Define search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    # Perform the search
    results = conn.search(
        collection_name=collection_name,
        data=[query_embedding],
        field="embedding",
        search_params=search_params,
        limit=limit,
        output_fields=["text", "mongo_id"]
    )

    return results[0]


async def find_relevant_videos(text, limit=5, duration=None) -> List[str]:
    try:
        # Step 1: Search for similar documents
        current_duration = 0
        results = await search_similar_documents(text, limit)

        if not results:
            logger.warning("No similar documents found")
            return []

        video_paths = []

        # Step 2: Process each result
        for result in results:
            if duration and current_duration > duration:
                break
            mongo_id = result.get("entity", {}).get("mongo_id")
            if not mongo_id:
                logger.warning("No mongo_id found in a search result")
                continue

            logger.info(f"Processing mongo_id: {mongo_id}")

            # Step 3: Retrieve image analysis from MongoDB
            image_analysis = mongo_db.get_image_analysis(mongo_id)
            if not image_analysis:
                logger.warning(
                    f"No image analysis found for mongo_id: {mongo_id}")
                continue

            # Step 4: Process the image analysis to find the relevant video
            metadata = image_analysis.get("metadata", {})
            video_path = metadata.get("clip_url")
            current_duration += image_analysis.get("fileData").get("duration")
            print("Debug: current_duration =", current_duration,
                  "duration =", duration)
            if not video_path:
                logger.warning(
                    f"No video path found in the image analysis metadata for mongo_id: {mongo_id}")
                continue

            full_video_path = video_path
            video_paths.append(full_video_path)
            logger.info(f"Found relevant video: {full_video_path}")

        logger.info(f"Total relevant videos found: {len(video_paths)}")
        return video_paths

    except Exception as e:
        logger.error(f"Error in find_relevant_videos: {str(e)}")
        return []


class VideoDescriptionWithTimes(BaseModel):
    description: str = Field(description="The description of the video scene")
    start_time: float = Field(
        description="The start time of the video in seconds")
    end_time: float = Field(description="The end time of the video in seconds")


class VideoDescription(BaseModel):
    data: List[VideoDescriptionWithTimes]


async def generate_video_descriptions(story: str, audio_duration: float) -> List[VideoDescriptionWithTimes]:
    try:

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        # Create a PydanticOutputParser
        parser = JsonOutputParser(
            pydantic_object=VideoDescription)
        logger.info("Generating video descriptions for the story")

        # model = genai.GenerativeModel('gemini-1.5-pro')
        # prompt = f"""
        # Given the following story and its audio duration, generate appropriate video scene descriptions that could be used to illustrate the story visually. Each description should be a single sentence and focus on a key moment or image from the story. The number and timing of scenes should be suitable for the audio duration, ensuring synchronization between the visuals and the narration.

        # Story:
        # {story}

        # Audio Duration: {audio_duration} seconds

        # Please provide a list of video scene descriptions, each on a new line. Consider the pacing of the story and the available time when determining the number and content of the scenes.

        # Video Scene Descriptions:
        # """

        # response = model.generate_content(prompt)

        # if response.text:
        #     # Split the response into individual descriptions
        #     descriptions = [d.strip()
        #                     for d in response.text.split('\n') if d.strip()]

        #     logger.info(f"Generated {len(descriptions)} video descriptions")
        #     print(descriptions)

        # Create a PydanticOutputParser

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template(
            """Given the following story and its audio duration, generate appropriate video scene descriptions that could be used to illustrate the story visually. Each description should be a single sentence and focus on a key moment or image from the story. The number and timing of scenes should be suitable for the audio duration, ensuring synchronization between the visuals and the narration.\n\nStory:\n{story}\n\nAudio Duration: {audio_duration} seconds\n\nPlease provide a list of video scene descriptions that could be used to illustrate the story visually."
            Story:
            {story}

            Audio Duration: {audio_duration} seconds

            Please provide a list of video scene description.
            {format_instructions}
            """
        )

        # Create the chain
        chain = prompt | llm | parser

        # Invoke the chain
        result = chain.invoke({
            "story": story,
            "audio_duration": audio_duration,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info(f"Video descriptions generated successfully")
        return result
    except Exception as e:
        logger.error(f"Error generating video descriptions: {str(e)}")
        raise


async def generate_video_descriptions_test(story: str, audio_duration: float) -> List[VideoDescriptionWithTimes]:
    return {'data': [{'description': 'A wide shot of a snow-covered Alaskan landscape with the Northern Lights shimmering in the sky.', 'start_time': 0, 'end_time': 8}, {'description': 'A map of the United States with Alaska highlighted, resembling an ice cream cone.', 'start_time': 8, 'end_time': 12}, {'description': 'A close-up shot of snow-capped mountains, emphasizing their immense size compared to the Himalayas.', 'start_time': 12, 'end_time': 20}, {'description': 'An aerial view of a massive glacier flowing between mountains, showcasing its slow movement.', 'start_time': 20, 'end_time': 28}, {'description': 'A brown bear hunts for salmon in a rushing river, highlighting its size and power.', 'start_time': 28, 'end_time': 38}, {'description': 'Underwater shot of bright red salmon swimming upstream, emphasizing their arduous journey.', 'start_time': 38, 'end_time': 45}, {'description': 'A bald eagle with its white head like a turban soars through the air, symbolizing freedom.', 'start_time': 45, 'end_time': 53}, {'description': 'A montage of various Alaskan landscapes, including glaciers, forests, and mountains, showcasing its stark beauty.', 'start_time': 53, 'end_time': 61}, {'description': 'A close-up shot of a delicate flower blooming in the snow, symbolizing the resilience of life.', 'start_time': 61, 'end_time': 68}, {'description': 'A wide shot of the Alaskan wilderness bathed in the golden light of sunset, evoking a sense of wonder.', 'start_time': 68, 'end_time': 76}, {'description': 'A person bundled up in warm clothes looks out at the snowy landscape, emphasizing the feeling of cold.', 'start_time': 76, 'end_time': 84}, {'description': "A final shot of the Northern Lights dancing across the sky, leaving a lasting impression of Alaska's beauty.", 'start_time': 84, 'end_time': 102}]}


def combine_media_segment(input_files: List[str], output_file: str, output_type: str = 'video', preserve_audio: bool = True):
    try:
        with open(f'file_list_{os.getpid()}.txt', 'w') as f:
            for file in input_files:
                f.write(f"file '{file}'\n")

        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', f'file_list_{os.getpid()}.txt',
        ]

        if output_type == 'video':
            if preserve_audio:
                cmd.extend(['-c:a', 'aac'])
            else:
                cmd.extend(['-an'])
            cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
        else:  # audio
            cmd.extend(['-c:a', 'aac'])

        cmd.append(output_file)

        subprocess.run(cmd, check=True, capture_output=True)
        return output_file
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error combining media segment: {e.stderr.decode()}")
    finally:
        if os.path.exists(f'file_list_{os.getpid()}.txt'):
            os.remove(f'file_list_{os.getpid()}.txt')


def combine_media_api(input_files: List[str], output_file: str, output_type: str = 'auto', preserve_audio: bool = True, max_workers: int = 4):
    try:
        print("Debug: input_files =", input_files)  # Add this line
        if not input_files:
            raise ValueError("No media files provided for combining.")

        if output_type == 'auto':
            output_type = Media(input_files[0]).media_type

        if output_type == 'video':
            output_file += '.mp4'
        else:  # audio
            output_file += '.mp3'

        # Split input files into chunks for parallel processing
        chunk_size = max(1, len(input_files) // max_workers)
        chunks = [input_files[i:i + chunk_size]
                  for i in range(0, len(input_files), chunk_size)]

        temp_outputs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                temp_output = f"temp_output_{i}.{output_type}"
                futures.append(executor.submit(
                    combine_media_segment, chunk, temp_output, output_type, preserve_audio))
                temp_outputs.append(temp_output)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        # Combine the temporary outputs
        final_combine_cmd = ['ffmpeg']
        for temp_output in temp_outputs:
            final_combine_cmd.extend(['-i', temp_output])
        final_combine_cmd.extend(['-filter_complex', f'concat=n={len(
            temp_outputs)}:v=1:a=1', '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', output_file])

        subprocess.run(final_combine_cmd, check=True, capture_output=True)

        # Clean up temporary files
        for temp_output in temp_outputs:
            os.remove(temp_output)

        return {"message": "Media combined successfully", "output_file": output_file}
    except Exception as e:
        raise ValueError(f"Error combining media: {str(e)}")


if __name__ == "__main__":
    # story = """
    # Imagine a place colder than the Himalayas, with days as short as a blink and nights longer than a Bollywood film.  This is Alaska, the largest state in America. Think of it as a giant ice cream cone on top of the US map! \n\nAlaska is a land of extremes. Towering mountains, bigger than any in India, pierce the sky.  These mountains are so tall, their peaks are always covered in snow, even in summer!  Imagine the Himalayas, but even bigger and with more snow.\n\nBetween these mountains flow giant rivers of ice called glaciers.  They are like frozen rivers, but much slower, moving just a little bit each year.  Some of these glaciers are bigger than entire cities in India!\n\nBut Alaska is not just ice and cold. It's also home to amazing animals.  Brown bears, bigger than any sloth bear you've ever seen, roam the forests and fish for salmon in the rivers.  These salmon, red like the sindoor worn by married women, travel thousands of miles to lay their eggs in these rivers.\n\nIn the sky, bald eagles soar on the wind.  These majestic birds, with their white heads that look like turbans, are a symbol of power and freedom.\n\nAlaska is a land of stark beauty and incredible wildlife.  It reminds us that even in the coldest and most remote corners of our planet, life finds a way. It also shows us the power of nature and how everything is connected.  From the glaciers to the salmon, to the bears and eagles, each part depends on the other.  \n\nSo, next time you feel a chill in the air, think of Alaska. It's a reminder that even in the coldest places, there is beauty and wonder to be found."""

    async def main():
        data = await generate_video_descriptions_test(
            "", 102)
        video_descriptions = data.get('data')
        video_segments = []
        i = 0
        for video_details in video_descriptions:
            i += 1
            description = video_details.get('description')
            start_time = video_details.get('start_time')
            end_time = video_details.get('end_time')
            # print(video_details)

            logger.info(f"Processing description {
                        i+1}/{len(video_descriptions)}")

            relevant_videos = await find_relevant_videos(description, 5, end_time - start_time)
            for video in relevant_videos:
                video_segments.append(video)
        #  combine videos
        combined_video = combine_media_api(
            input_files=video_segments,
            output_file="combined_video.mp4",
            output_type='video'
        )
        print("Debug: relevant_videos =", relevant_videos)  # Add this line
        logger.info(f"Found relevant video for description {i+1}")

        #  write the combined video to a file
        # with open("combined_video.mp4", "wb") as out:
        #     out.write(combined_video)

        # # Trim the video to match the audio segment
        # print(relevant_videos)
        # combined_video = combine_media_api(
        #     input_files=relevant_videos,
        #     output_file="combined_video.mp4",
        #     output_type='video'
        # )
        # start_time = 0
        # end_time = end_time * 1000
        # trimmed_video = f"trimmed_video_{i}.mp4"
        # trim_media_api(combined_video, trimmed_video, start_time, end_time)

        # logger.info(f"Trimmed video for segment {i+1}")

    asyncio.run(main())
