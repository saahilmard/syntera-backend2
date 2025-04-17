from fastapi import FastAPI, UploadFile, File
from whisper import load_model
from rag import generate_soap_notes
from reverse_rag import extract_and_verify
import shutil

app = FastAPI()
whisper_model = load_model("base")

@app.get("/")
async def root():
    return {"message": "Welcome to Syntera API"}

@app.post("/generate_notes/")
async def generate_notes(audio: UploadFile = File(...), patient_age: int = 0, visit_type: str = "general"):
    audio_file = "temp_audio.wav"
    with open(audio_file, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    transcription = whisper_model.transcribe(audio_file)['text']
    soap_notes = generate_soap_notes(transcription, patient_age, visit_type)
    verified_notes = extract_and_verify(soap_notes, transcription)

    return {
        "transcription": transcription,
        "verified_soap_notes": verified_notes
    }
