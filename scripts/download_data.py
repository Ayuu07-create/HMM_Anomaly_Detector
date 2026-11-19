from pathlib import Path
import io, zipfile, urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/"data"/"raw"/"MetroPT3(AirCompressor).csv"
TARGET.parent.mkdir(parents=True, exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"

print("Downloading MetroPT-3 from UCI...")
data = urllib.request.urlopen(URL, timeout=60).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    members = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not members:
        raise RuntimeError("No CSV found in UCI archive.")
    with z.open(members[0]) as src, TARGET.open("wb") as dst:
        dst.write(src.read())
print(f"Dataset saved to: {TARGET}")
