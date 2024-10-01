from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility


class ObjectLabel(BaseModel):
    name: str
    confidence: float = Field(...,
                              description="Confidence level of the label, between 0 and 1.")


class DominantColor(BaseModel):
    color: str = Field(...,
                       description="Name of the dominant color in the image in english")
    percentage: float = Field(...,
                              description="Percentage of the color in the image.")


class FaceDetection(BaseModel):
    number_of_faces: int = Field(...,
                                 description="Number of faces detected in the image.")
    face_details: Optional[List[str]] = Field(
        None, description="Descriptions of facial features like expressions, age, etc.")


class EmotionDetection(BaseModel):
    emotion: str
    confidence: float = Field(
        ..., description="Confidence level of the detected emotion, between 0 and 1.")


class SceneTransition(BaseModel):
    probability: float = Field(
        ..., description="Likelihood of scene transition if part of a video frame.")


class Metadata(BaseModel):
    title: str = Field(
        ..., description="Title of the image.")
    clip_id: str = Field(
        ..., description="ID of the image in the clip.")
    frame_number: int = Field(
        ..., description="Frame number of the image in the clip.")
    frame_timestamp_percentage: float = Field(
        ..., description="Timestamp of the image in the clip as a percentage of the total duration.")
    clip_url: str = Field(
        ..., description="URL of the clip containing the image.")


class ImageAnalysis(BaseModel):
    image_url: str = Field(..., description="URL of the image to analyze.")
    metadata: Metadata = Field(
        ..., description="Metadata of the image.")
    tags: List[str] = Field(
        ..., description="List of relevant tags describing the elements of the image.")
    scene_type: str = Field(
        ..., description="Type of scene, such as landscape, portrait, still life, etc.")
    dominant_colors: List[DominantColor] = Field(
        ..., description="List of dominant colors with their percentages in the image.")
    motion_intensity: str = Field(
        ..., description="Level of motion in the image: low, medium, or high.")
    object_labels: List[ObjectLabel] = Field(
        ..., description="Main objects or elements visible in the image.")
    facial_detection: Optional[FaceDetection] = Field(
        None, description="Details about any detected faces in the image.")
    text_overlay: Optional[str] = Field(
        None, description="Text that appears over the image, if any.")
    action_type: str = Field(...,
                             description="Indicates if the scene is static or dynamic.")
    time_of_day: Optional[str] = Field(
        None, description="Approximate time of day depicted in the image")
    weather_conditions: Optional[str] = Field(
        None, description="Weather conditions in the image")
    camera_angle: Optional[str] = Field(
        None, description="Perspective or angle from which the image was taken.")
    transcript: Optional[str] = Field(
        None, description="If the image is a video frame, the transcribed speech in the frame.")
    emotion_detection: Optional[List[EmotionDetection]] = Field(
        None, description="List of detected emotions in the image, if people are present.")
    scene_transition_probability: Optional[SceneTransition] = Field(
        None, description="Likelihood of scene change in a video frame.")
    detailed_description: str = Field(
        ..., description="A detailed description of the image. dont consider the text overlay")
    keywords: List[str] = Field(...,
                                description="List of keywords that best represent the image.")
    last_analyzed: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of when the image was last analyzed.")


def create_milvus_collection():
    try:
        # Connect to Milvus
        conn = connections.connect(host="127.0.0.1", port=19530)

        collection_name = "image_analysis_2"

        # Check if the collection exists
        if not utility.has_collection(collection_name):
            # Define a schema for the collection (including MongoDB ID as a string)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64,
                            is_primary=True, auto_id=True),
                FieldSchema(name="embedding",
                            dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR,
                            max_length=10000),
                FieldSchema(name="mongo_id",
                            dtype=DataType.VARCHAR, max_length=50)
            ]

            # Define the schema for the collection
            schema = CollectionSchema(
                fields, description="Image embeddings collection with raw text and MongoDB ID")

            # Create the collection
            collection = Collection(name=collection_name, schema=schema)
            print(f"Collection '{collection_name}' created.")
        else:
            collection = Collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists.")

        # Create an index if it doesn't exist
        if not collection.has_index():
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="embedding", index_params=index_params)
            print(f"Index created for collection '{collection_name}'.")
        else:
            print(f"Index already exists for collection '{collection_name}'.")

        return collection
    except Exception as e:
        print(f"Error creating or accessing Milvus collection: {e}")
        return None
