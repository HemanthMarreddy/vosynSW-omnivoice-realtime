import time
import os
import torch
import torchaudio

from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"
OUTPUT = "outputs/smoke_test.wav"

TEXT = "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทยแบบเรียลไทม์"


print("=" * 60)
print("OmniVoice Thai Smoke Test")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

os.makedirs("outputs", exist_ok=True)

print("\nLoading model...")

load_start = time.perf_counter()

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)

load_end = time.perf_counter()

print(f"Model load time: {load_end - load_start:.3f} sec")

print("\nGenerating Thai audio...")

generation_start = time.perf_counter()

audio = model.generate(
    text=TEXT,
)

generation_end = time.perf_counter()

generation_time = generation_end - generation_start

print(f"Generation time: {generation_time:.3f} sec")

waveform = audio[0]

torchaudio.save(
    OUTPUT,
    waveform,
    24000,
)

duration = waveform.shape[-1] / 24000

print(f"Audio duration: {duration:.3f} sec")
print(f"RTF: {generation_time / duration:.4f}")
print(f"Saved: {OUTPUT}")

print("\nSmoke test completed.")
