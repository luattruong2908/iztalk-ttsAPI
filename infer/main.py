import os
import gradio as gr
import uuid
import shutil
from f5tts_wrapper import F5TTSWrapper

# Cấu hình môi trường <3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUTF8"] = "1"

# Thư mục model
model_dir = "model"
model_path = os.path.join(model_dir, "model_612000.safetensors")
vocab_path = os.path.join(model_dir, "vocab.txt")

# Tạo thư mục output
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
tts = F5TTSWrapper(
    vocoder_name="vocos",
    ckpt_path=model_path,
    vocab_file=vocab_path,
    use_ema=False,
)

# Hàm xử lý
def clone(text, audio_file, ref_text=""):
    # Lưu file audio mẫu tạm thời
    ext = audio_file.split(".")[-1]
    ref_audio_path = os.path.join(TEMP_DIR, f"ref_{uuid.uuid4()}.{ext}")
    shutil.copy(audio_file, ref_audio_path)

    # Xử lý giọng tham chiếu
    tts.preprocess_reference(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        clip_short=True
    )

    # Sinh giọng mới
    output_path = os.path.join(OUTPUT_DIR, f"output_{uuid.uuid4()}.wav")
    tts.generate(
        text=text,
        output_path=output_path,
        nfe_step=20,
        cfg_strength=2.0,
        speed=1.0,
        cross_fade_duration=-0
    )

    return output_path

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=clone,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Audio(type="filepath", label="Voice reference audio file"),
        gr.Textbox(label="Reference text (optional)", lines=2, placeholder="Optional text spoken in reference audio")
    ],
    outputs=gr.Audio(type="filepath"),
    title="izTalk Voice Cloner",
    description="Clone voice uizTalk model with reference audio.",
    examples=[
        ["Xin chào, tôi là một bậc thầy nhái giọng, hãy viết vào đây những gì bạn muốn tôi nói.", "./ref_sample/midside_woman_ref.wav", "Thậm chí không ăn thì cũng có cảm giác rất là cứng bụng, chủ yếu là cái phần rốn...trở lên. Em có cảm giác khó thở, và ngủ cũng không ngon, thường bị ợ hơi rất là nhiều"],
        ["Chào bạn, mình là Sơn Tùng em ti pi. Có muốn làm kiểu ảnh không nào?", "./ref_sample/SonTungMTP.mp3", ""]
    ]
)

iface.launch()
