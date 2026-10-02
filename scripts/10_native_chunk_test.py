import time
from pathlib import Path

import torch
import torchaudio

from omnivoice import OmniVoice, OmniVoiceGenerationConfig


MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIR = Path("outputs/native_chunk_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT = (
    "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย "
    "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์ "
    "โดยให้ความสำคัญกับคุณภาพของเสียงและเวลาในการตอบสนอง "
    "การทดสอบนี้ใช้ข้อความภาษาไทยต่อเนื่องเพื่อประเมินการสร้างเสียงแบบแบ่งช่วง"
)

print("Loading OmniVoice...")

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
)

print("Model loaded.")

configs = {
    "native_15s": OmniVoiceGenerationConfig(
        audio_chunk_duration=15.0,
        audio_chunk_threshold=0.0,
    ),

    "native_8s": OmniVoiceGenerationConfig(
        audio_chunk_duration=8.0,
        audio_chunk_threshold=0.0,
    ),

    "native_5s": OmniVoiceGenerationConfig(
        audio_chunk_duration=5.0,
        audio_chunk_threshold=0.0,
    ),
}


for name, config in configs.items():

    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print(f"Chunk duration: {config.audio_chunk_duration}s")
    print(f"Chunk threshold: {config.audio_chunk_threshold}s")

    start = time.perf_counter()

    try:
        audios = model.generate(
            text=TEXT,
            language="Thai",
            generation_config=config,
        )

        elapsed = time.perf_counter() - start

        waveform = audios[0]

        if waveform.numel() == 0:
            print("STATUS: EMPTY AUDIO")
            continue

        duration = waveform.shape[-1] / 24000
        rtf = elapsed / duration

        output = OUTPUT_DIR / f"{name}.wav"

        torchaudio.save(
            str(output),
            waveform,
            24000,
        )

        print(f"STATUS: SUCCESS")
        print(f"Total latency: {elapsed:.3f}s")
        print(f"Audio duration: {duration:.3f}s")
        print(f"RTF: {rtf:.3f}")
        print(f"Output: {output}")

    except Exception as e:

        print("STATUS: FAILED")
        print(type(e).__name__, str(e))


print("\nAll native chunk tests completed.")
