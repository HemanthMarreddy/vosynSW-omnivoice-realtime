import time
from pathlib import Path

import torch
import torchaudio

from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIR = Path(
    "/home/hemanth_marreddy99/sprint53/omnivoice-realtime/"
    "outputs/manual_streaming_test"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CHUNKS = [
    "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย",
    "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์",
    "โดยให้ความสำคัญกับคุณภาพของเสียงและเวลาในการตอบสนอง",
    "เราต้องการตรวจสอบว่าระบบสามารถสร้างและส่งเสียงออกมาอย่างต่อเนื่องได้หรือไม่",
]


print("=" * 70)
print("OmniVoice Thai Manual Incremental Streaming Test")
print("=" * 70)

print("\nLoading model...")

model = OmniVoice.from_pretrained(
    MODEL_ID,
)

print("Model loaded.")

all_results = []

test_start = time.perf_counter()

for i, text in enumerate(CHUNKS, 1):

    print("\n" + "=" * 70)
    print(f"CHUNK {i}")
    print("=" * 70)

    print(f"Text: {text}")
    print(f"Characters: {len(text)}")

    start = time.perf_counter()

    try:

        with torch.inference_mode():

            audio = model.generate(
                text=text,
                language="Thai",
            )[0]

        generation_time = time.perf_counter() - start

        if audio is None or audio.numel() == 0:
            print("FAILED: Empty audio")
            continue

        if audio.dim() == 2:
            audio_duration = audio.shape[-1] / model.sampling_rate
        else:
            audio_duration = audio.shape[0] / model.sampling_rate

        rtf = (
            generation_time / audio_duration
            if audio_duration > 0
            else float("inf")
        )

        arrival_time = time.perf_counter() - test_start

        output_path = OUTPUT_DIR / f"chunk_{i}.wav"

        audio_to_save = audio.detach().cpu()

        if audio_to_save.dim() == 1:
            audio_to_save = audio_to_save.unsqueeze(0)

        torchaudio.save(
            str(output_path),
            audio_to_save,
            model.sampling_rate,
        )

        print(f"STATUS             : SUCCESS")
        print(f"Generation latency : {generation_time:.3f} s")
        print(f"Audio duration     : {audio_duration:.3f} s")
        print(f"RTF                : {rtf:.3f}")
        print(f"Arrival time       : {arrival_time:.3f} s")
        print(f"Saved              : {output_path}")

        all_results.append({
            "chunk": i,
            "latency": generation_time,
            "duration": audio_duration,
            "rtf": rtf,
            "arrival": arrival_time,
        })

    except Exception as e:

        print("STATUS             : FAILED")
        print(f"ERROR              : {type(e).__name__}: {e}")


print("\n")
print("=" * 70)
print("FINAL STREAMING FEASIBILITY SUMMARY")
print("=" * 70)

if not all_results:

    print("No successful chunks.")

else:

    total_wall_time = time.perf_counter() - test_start

    total_audio = sum(
        r["duration"]
        for r in all_results
    )

    total_generation = sum(
        r["latency"]
        for r in all_results
    )

    overall_rtf = (
        total_generation / total_audio
        if total_audio > 0
        else float("inf")
    )

    print(f"Successful chunks : {len(all_results)}")
    print(f"Total audio       : {total_audio:.3f} s")
    print(f"Total generation  : {total_generation:.3f} s")
    print(f"Wall time         : {total_wall_time:.3f} s")
    print(f"Overall RTF       : {overall_rtf:.3f}")

    print("\nPer-chunk results:")

    for r in all_results:

        print(
            f"Chunk {r['chunk']}: "
            f"latency={r['latency']:.3f}s | "
            f"audio={r['duration']:.3f}s | "
            f"RTF={r['rtf']:.3f} | "
            f"arrival={r['arrival']:.3f}s"
        )

    print("\nInterpretation:")

    first = all_results[0]

    if first["latency"] < first["duration"]:
        print("First chunk generation is faster than playback duration.")
    else:
        print("First chunk generation is slower than playback duration.")

    if all(r["rtf"] < 1.0 for r in all_results):
        print("All chunks meet real-time generation speed.")
    else:
        print("One or more chunks are slower than real-time generation.")
