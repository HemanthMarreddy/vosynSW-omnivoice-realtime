import time
from pathlib import Path

import torch
import torchaudio

from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIR = Path("outputs/chunked")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


chunks = [
    "สวัสดีครับ",
    "นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย",
    "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์",
]


print("Loading model...")

load_start = time.perf_counter()

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)

load_end = time.perf_counter()

print(f"Load time: {load_end - load_start:.3f}s")


results = []

for i, text in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"CHUNK {i}")
    print(text)

    start = time.perf_counter()

    audio = model.generate(
        text=text,
    )

    end = time.perf_counter()

    elapsed = end - start

    waveform = audio[0]

    duration = waveform.shape[-1] / 24000

    rtf = elapsed / duration if duration > 0 else None

    output = OUTPUT_DIR / f"chunk_{i:02d}.wav"

    torchaudio.save(
        str(output),
        waveform,
        24000,
    )

    results.append(
        {
            "chunk": i,
            "latency": elapsed,
            "duration": duration,
            "rtf": rtf,
            "output": str(output),
        }
    )

    print(f"Latency: {elapsed:.3f}s")
    print(f"Duration: {duration:.3f}s")
    print(f"RTF: {rtf:.4f}")
    print(f"Output: {output}")


print("\nFINAL RESULTS")

for r in results:
    print(r)
