import csv
from pathlib import Path


OUTPUT = Path("results/omnivoice_streaming_smoke.csv")

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


rows = [
    {
        "model": "OmniVoice-Streaming",
        "language": "Thai",
        "test": "chunked_smoke",
        "sample": "smoke_001",
        "ttfa_sec": "",
        "latency_sec": "",
        "audio_duration_sec": "",
        "rtf": "",
        "cer": "",
        "quality_observation": "",
        "status": "",
    }
]


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


print(f"Created {OUTPUT}")
