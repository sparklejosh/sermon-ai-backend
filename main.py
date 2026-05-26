from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import shutil
import os

app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")

@app.get("/")
def home():
    return {"status": "Sermon AI Running on Render"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    path = "audio.mp3"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, info = model.transcribe(path)

    text = ""
    for s in segments:
        text += s.text + " "

    return {"transcript": text}