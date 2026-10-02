from pathlib import Path
import csv


BASE = Path("data/flores")
OUTPUT = BASE / "thai_benchmark.csv"

FILES = [
    BASE / "tha_Thai.dev",
    BASE / "tha_Thai.devtest",
]


rows = []

sample_id = 0

for file_path in FILES:

    if not file_path.exists():
        print(f"WARNING: missing {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            text = line.strip()

            if not text:
                continue

            sample_id += 1

            rows.append({
                "sample_id": f"thai_{sample_id:04d}",
                "source_file": file_path.name,
                "line_number": line_number,
                "text": text,
            })


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_id",
            "source_file",
            "line_number",
            "text",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print(f"Created: {OUTPUT}")
print(f"Total samples: {len(rows)}")
