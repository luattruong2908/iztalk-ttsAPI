import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUTF8"] = "1"

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import uuid
from f5tts_wrapper import F5TTSWrapper

# Init FastAPI
app = FastAPI()

# Create output directories
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model on startup
tts = F5TTSWrapper(
    vocoder_name="vocos",
    ckpt_path="model/model_612000.safetensors",
    vocab_file="model/vocab.txt",
    use_ema=False,
)

@app.post("/synthesize")
async def synthesize(
    input_text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form("")
):
    try:
        # Save reference audio file
        ref_audio_ext = ref_audio.filename.split(".")[-1]
        ref_audio_path = os.path.join(TEMP_DIR, f"ref_{uuid.uuid4()}.{ref_audio_ext}")
        with open(ref_audio_path, "wb") as buffer:
            shutil.copyfileobj(ref_audio.file, buffer)

        # Preprocess reference audio
        tts.preprocess_reference(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            clip_short=True
        )

        # Generate audio
        output_path = os.path.join(OUTPUT_DIR, f"generated_{uuid.uuid4()}.wav")
        tts.generate(
            text=input_text,
            output_path=output_path,
            nfe_step=20,
            cfg_strength=2.0,
            speed=1.0,
            cross_fade_duration=0.15
        )

        return FileResponse(output_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
