import asyncio
import time
from pathlib import Path

import torch
import torchaudio

import omnivoice.openai_tts_server as server


TEXT = (
    "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย "
    "เรากำลังตรวจสอบความสามารถในการสร้างเสียงแบบเรียลไทม์ "
    "โดยให้ความสำคัญกับคุณภาพของเสียงและเวลาในการตอบสนอง "
    "การทดสอบนี้ใช้ข้อความภาษาไทยต่อเนื่องเพื่อประเมินประสิทธิภาพของระบบ "
    "เราต้องการตรวจสอบว่าระบบสามารถสร้างเสียงและส่งผลลัพธ์ออกมาเป็นช่วงๆ "
    "ได้อย่างต่อเนื่องหรือไม่ "
    "นอกจากนี้เรายังต้องการวัดเวลาในการสร้างเสียงแต่ละช่วง "
    "และเปรียบเทียบเวลาที่ใช้กับความยาวของเสียงที่สร้างขึ้น "
    "เพื่อประเมินความเป็นไปได้ของการสังเคราะห์เสียงแบบเรียลไทม์ "
    "บนเครื่องประมวลผลแบบซีพียู "
    "หากระบบสามารถสร้างเสียงแต่ละช่วงได้เร็วกว่าหรือใกล้เคียงกับ "
    "ระยะเวลาของเสียงที่สร้างขึ้น ระบบก็อาจสามารถนำไปใช้กับ "
    "การส่งเสียงแบบต่อเนื่องได้"
)

OUTPUT_DIR = Path(
    "/home/hemanth_marreddy99/sprint53/omnivoice-realtime/"
    "outputs/internal_chunk_timing"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def main():

    print("=" * 70)
    print("OmniVoice Internal Chunk Timing Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # Initialize the same locks used by the real FastAPI server
    # ---------------------------------------------------------

    server.service.set_lock(asyncio.Lock())
    server.service.set_generation_lock(asyncio.Semaphore(1))
    server.asr_service.set_lock(asyncio.Lock())
    server.secondary_worker_manager.set_lock(asyncio.Lock())

    print("\nServer locks initialized.")

    # ---------------------------------------------------------
    # Build request
    # ---------------------------------------------------------

    payload = server.SpeechRequest(
        text=TEXT,
        model=server.API_MODEL_ID,
        voice="alloy",
        response_format="wav",
        language="Thai",

        # We want to test the chunk pipeline itself.
        sentence_chunking=False,

        # Force audio chunking.
        audio_chunk_duration=5.0,
        audio_chunk_threshold=0.0,
    )

    print("\nPreparing request...")

    prepared = server._prepare_request(payload)

    print("Prepared successfully.")
    print(f"Full text characters: {len(prepared.text)}")
    print(f"Planned chunks: {len(prepared.chunk_plan)}")

    for i, text in enumerate(prepared.chunk_plan, 1):
        print(f"\nChunk plan {i}:")
        print(text)

    # ---------------------------------------------------------
    # Build TextChunk objects
    # ---------------------------------------------------------

    chunks = server._build_text_chunks(prepared.chunk_plan)

    print("\nActual chunks:")
    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_id}: "
            f"{len(chunk.text)} characters"
        )

    # ---------------------------------------------------------
    # Single-worker measurement
    # ---------------------------------------------------------

    print("\nUsing SINGLE worker for clean CPU measurement.")

    request_id = f"timing-{int(time.time())}"

    print(f"Request ID: {request_id}")

    test_start = time.perf_counter()

    results = []
    arrival_times = {}

    # ---------------------------------------------------------
    # Run internal chunk pipeline
    # ---------------------------------------------------------

    async for result in server._iter_ordered_chunk_results(
        chunks,
        prepared,
        payload,
        request_id=request_id,
        secondary_available=False,
    ):

        arrival_time = time.perf_counter() - test_start

        results.append(result)
        arrival_times[result.chunk_id] = arrival_time

        audio_duration = (
            result.waveform.shape[-1]
            / result.sample_rate
        )

        chunk_rtf = (
            result.latency_s / audio_duration
            if audio_duration > 0
            else float("inf")
        )

        print("\n" + "=" * 70)
        print(f"CHUNK {result.chunk_id} AVAILABLE")
        print("=" * 70)

        print(
            f"Generation latency : "
            f"{result.latency_s:.3f} s"
        )

        print(
            f"Audio duration     : "
            f"{audio_duration:.3f} s"
        )

        print(
            f"Chunk RTF          : "
            f"{chunk_rtf:.3f}"
        )

        print(
            f"Arrival from start : "
            f"{arrival_time:.3f} s"
        )

        if len(results) == 1:

            print(
                f"TTFA               : "
                f"{arrival_time:.3f} s"
            )

        else:

            previous = arrival_times[
                results[-2].chunk_id
            ]

            print(
                f"Inter-chunk arrival: "
                f"{arrival_time - previous:.3f} s"
            )

        # -----------------------------------------------------
        # Save chunk
        # -----------------------------------------------------

        output = (
            OUTPUT_DIR /
            f"chunk_{result.chunk_id}.wav"
        )

        torchaudio.save(
            str(output),
            result.waveform.cpu(),
            result.sample_rate,
        )

        print(f"Saved              : {output}")

    total_time = time.perf_counter() - test_start

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total_audio = sum(
        r.waveform.shape[-1] / r.sample_rate
        for r in results
    )

    total_model_time = sum(
        r.latency_s
        for r in results
    )

    overall_rtf = (
        total_time / total_audio
        if total_audio > 0
        else float("inf")
    )

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Chunks generated      : {len(results)}")
    print(f"Total wall time       : {total_time:.3f} s")
    print(f"Total model time      : {total_model_time:.3f} s")
    print(f"Total audio duration  : {total_audio:.3f} s")
    print(f"Overall RTF           : {overall_rtf:.3f}")

    if results:
        print(
            f"TTFA                  : "
            f"{arrival_times[results[0].chunk_id]:.3f} s"
        )

    print("\nPer-chunk results:")

    for result in results:

        duration = (
            result.waveform.shape[-1]
            / result.sample_rate
        )

        rtf = (
            result.latency_s / duration
            if duration > 0
            else float("inf")
        )

        arrival = arrival_times[result.chunk_id]

        print(
            f"Chunk {result.chunk_id}: "
            f"latency={result.latency_s:.3f}s | "
            f"audio={duration:.3f}s | "
            f"RTF={rtf:.3f} | "
            f"arrival={arrival:.3f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
