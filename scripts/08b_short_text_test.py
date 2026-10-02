import time
import random
import numpy as np
import torch
import torchaudio
from pathlib import Path

from omnivoice import OmniVoice


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIR = Path("outputs/short_text_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tests = {
    "very_short": "สวัสดีครับ",

    "short": "สวัสดีครับ นี่คือการทดสอบ",

    "medium": "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย",

    "long": (
        "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย "
        "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์ "
        "โดยให้ความสำคัญกับคุณภาพของเสียงและเวลาในการตอบสนอง"
    ),
}


print("Loading OmniVoice...")

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)

print("Model loaded.\n")


for name, text in tests.items():

    print("=" * 70)
    print(name)
    print(text)

    # Reset deterministic state
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start = time.perf_counter()

    try:
        audio = model.generate(text=text)

        elapsed = time.perf_counter() - start

        waveform = audio[0]

        if waveform.numel() == 0:
            print("STATUS: EMPTY AUDIO")
            continue

        duration = waveform.shape[-1] / 24000
        rtf = elapsed / duration if duration > 0 else None

        output = OUTPUT_DIR / f"{name}.wav"

        torchaudio.save(
            str(output),
            waveform,
            24000,
        )

        print(f"STATUS: SUCCESS")
        print(f"Latency: {elapsed:.3f}s")
        print(f"Duration: {duration:.3f}s")
        print(f"RTF: {rtf:.3f}")
        print(f"Output: {output}")

    except Exception as e:

        elapsed = time.perf_counter() - start

        print("STATUS: FAILED")
        print(f"Latency before failure: {elapsed:.3f}s")
        print(f"Error: {type(e).__name__}: {e}")


print("\nTest completed.")
