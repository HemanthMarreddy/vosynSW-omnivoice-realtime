import csv
from pathlib import Path


INPUT = Path("data/flores/thai_benchmark.csv")
OUTPUT = Path("data/flores/smoke_sample.csv")


with open(INPUT, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


if not rows:
    raise RuntimeError("No Thai FLORES samples found.")


sample = rows[0]


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=sample.keys(),
    )

    writer.writeheader()
    writer.writerow(sample)


print("Smoke-test sample:")
print(sample)

print(f"\nSaved: {OUTPUT}")
