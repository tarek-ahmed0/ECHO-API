from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import speech_recognition as sr

# Our Models :
from text_preprocessing import get_original_text
from translation_pipeline import coll2formal_translation
from sentiment_analysis_pipeline import sentiment_analysis

app = FastAPI()

# ==================== VIDEO API ====================
@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = get_original_text(temp_filename)
        if not isinstance(extracted_text, str) or extracted_text.startswith("Error"):
            os.remove(temp_filename)
            return {"error": "Failed to extract text from video", "details": extracted_text}

        transformed_text = coll2formal_translation(extracted_text)
        if not isinstance(transformed_text, str):
            os.remove(temp_filename)
            return {"error": "Translation failed"}

        try:
            words, signids = sentiment_analysis(transformed_text)
        except Exception as e:
            os.remove(temp_filename)
            return {"error": "Sentiment analysis failed", "details": str(e)}

        os.remove(temp_filename)
        return {
            "Video Original Text": extracted_text,
            "Transformed Text": transformed_text,
            "The Nearest Words Of Stored": words,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected server error", "details": str(e)},
        )


# ==================== AUDIO API ====================
def extract_text(file_path: str) -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language="ar-EG")
    except sr.UnknownValueError:
        return "error: Could not understand the audio (unclear or silent)"
    except sr.RequestError as e:
        return f"error: Google Speech Recognition API error: {e}"
    except Exception as e:
        return f"error: {str(e)}"

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.wav"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        extracted_text = extract_text(temp_path)

        if extracted_text.startswith("error:"):
            return {"error": "Failed to extract text", "details": extracted_text}

        transformed_text = coll2formal_translation(extracted_text)
        if not isinstance(transformed_text, str) or transformed_text.strip() == "":
            return {"error": "Translation failed"}

        try:
            words, signids = sentiment_analysis(transformed_text)
        except Exception as e:
            return {"error": "Sentiment analysis failed", "details": str(e)}

        return {
            "Audio Original Text": extracted_text,
            "Transformed Text": transformed_text,
            "The Nearest Words Of Stored": words,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
