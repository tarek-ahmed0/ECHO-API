from moviepy import VideoFileClip
import speech_recognition as sr
import tempfile
import os

def get_original_text(video_path):
    try:
        if not os.path.exists(video_path):
            return "Error: Video file not found."

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_path = tmp_audio_file.name

        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                return "Error: Video has no audio track."
            
            video.audio.write_audiofile(tmp_audio_path, fps=16000, codec='pcm_s16le')
            video.close()  # Ensure the video is closed after audio extraction

        except Exception as e:
            return f"Error during audio extraction: {str(e)}"

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_audio_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language="ar-AR")
        os.remove(tmp_audio_path) 
        return text

    except Exception as e:
        return f"Unexpected error: {str(e)}"
