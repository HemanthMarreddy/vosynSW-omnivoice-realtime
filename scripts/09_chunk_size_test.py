import time
from pathlib import Path

import torchaudio

from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIR = Path("outputs/chunk_size_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


tests = {
    "short": [
        "สวัสดีครับ"
    ],

    "medium": [
        "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย"
    ],

    "long": [
        "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย "
        "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์ "
        "โดยให้ความสำคัญกับคุณภาพของเสียงและเวลาในการตอบสนอง"
    ],
}


print("Loading model...")

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)


for name, chunks in tests.items():

    text = " ".join(chunks)

    print("\n" + "=" * 70)
    print(name.upper())
    print(text)

    start = time.perf_counter()

    audio = model.generate(
        text=text,
    )

    elapsed = time.perf_counter() - start

    waveform = audio[0]

    duration = waveform.shape[-1] / 24000

    rtf = elapsed / duration if duration > 0 else None

    output = OUTPUT_DIR / f"{name}.wav"

    torchaudio.save(
        str(output),
        waveform,
        24000,
    )

    print(f"Latency: {elapsed:.3f}s")
    print(f"Duration: {duration:.3f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"Output: {output}")
