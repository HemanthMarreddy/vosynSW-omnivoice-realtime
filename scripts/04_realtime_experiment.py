import csv
import time
from pathlib import Path

import torch
import torchaudio

from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

INPUT = Path("data/flores/smoke_sample.csv")
OUTPUT_DIR = Path("outputs/realtime_smoke")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(INPUT, "r", encoding="utf-8") as f:
    row = next(csv.DictReader(f))


text = row["text"]

print("=" * 70)
print("OMNIVOICE THAI REAL-TIME FEASIBILITY TEST")
print("=" * 70)

print("\nText:")
print(text)

print("\nLoading model...")

load_start = time.perf_counter()

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)

load_end = time.perf_counter()

print(f"Model load time: {load_end - load_start:.3f}s")


print("\nStarting generation...")

generation_start = time.perf_counter()

audio = model.generate(
    text=text,
)

generation_end = time.perf_counter()

total_generation_time = generation_end - generation_start

waveform = audio[0]

sample_rate = 24000

duration = waveform.shape[-1] / sample_rate

output = OUTPUT_DIR / f"{row['sample_id']}.wav"

torchaudio.save(
    str(output),
    waveform,
    sample_rate,
)


rtf = total_generation_time / duration if duration > 0 else None


print("\nRESULT")
print("-" * 70)

print(f"Sample ID:              {row['sample_id']}")
print(f"Audio duration:         {duration:.3f}s")
print(f"Generation time:        {total_generation_time:.3f}s")
print(f"RTF:                    {rtf:.4f}")
print(f"Output:                 {output}")

print("\nIMPORTANT:")
print(
    "This model.generate() measurement is completion latency, "
    "not proof of true incremental streaming."
)
