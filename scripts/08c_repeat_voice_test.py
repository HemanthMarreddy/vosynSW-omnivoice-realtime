import time
from pathlib import Path

import torch
import torchaudio

from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

TEXT = (
    "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย "
    "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์"
)

OUTPUT_DIR = Path("outputs/repeat_voice_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)


for i in range(1, 4):

    print(f"\nGenerating repetition {i}...")

    start = time.perf_counter()

    audio = model.generate(
        text=TEXT,
    )

    elapsed = time.perf_counter() - start

    waveform = audio[0]

    if waveform.numel() == 0:
        print("EMPTY AUDIO")
        continue

    duration = waveform.shape[-1] / 24000

    output = OUTPUT_DIR / f"repeat_{i}.wav"

    torchaudio.save(
        str(output),
        waveform,
        24000,
    )

    print(f"Latency: {elapsed:.3f}s")
    print(f"Duration: {duration:.3f}s")
    print(f"RTF: {elapsed / duration:.3f}")
    print(f"Saved: {output}")
