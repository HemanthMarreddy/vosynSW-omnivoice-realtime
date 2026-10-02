import time
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

from omnivoice import OmniVoice


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIR = Path("outputs/deterministic_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


chunks = [
    "สวัสดีครับ",
    "นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย",
    "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์",
]


print("Loading model...")

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)


for i, text in enumerate(chunks, start=1):

    # Reset seed before each independent generation
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n" + "=" * 60)
    print(f"Chunk {i}")
    print(text)

    start = time.perf_counter()

    audio = model.generate(
        text=text,
    )

    elapsed = time.perf_counter() - start

    waveform = audio[0]

    duration = waveform.shape[-1] / 24000

    rtf = elapsed / duration if duration > 0 else None

    output = OUTPUT_DIR / f"chunk_{i:02d}.wav"

    torchaudio.save(
        str(output),
        waveform,
        24000,
    )

    print(f"Latency: {elapsed:.3f}s")
    print(f"Duration: {duration:.3f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"Output: {output}")
