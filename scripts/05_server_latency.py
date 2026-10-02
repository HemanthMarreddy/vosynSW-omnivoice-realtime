import time
import requests
from pathlib import Path


URL = "http://127.0.0.1:6655/v1/audio/speech"

TEXT = "สวัสดีครับ นี่คือการทดสอบเวลาเริ่มต้นของระบบสังเคราะห์เสียงภาษาไทย"

OUTPUT = Path("outputs/server_latency_test.mp3")


payload = {
    "model": "omnivoice",
    "voice": "nova",
    "input": TEXT,
    "response_format": "mp3",
}


start = time.perf_counter()

response = requests.post(
    URL,
    json=payload,
    timeout=300,
)

end = time.perf_counter()

response.raise_for_status()

OUTPUT.write_bytes(response.content)

latency = end - start

print(f"HTTP request completed in: {latency:.3f}s")
print(f"Output size: {len(response.content)} bytes")
print(f"Saved: {OUTPUT}")
