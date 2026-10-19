import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path("data/raw")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)


def sha256_of(path: Path) -> str:
    """Hash in 1 MB chunks so large files never sit fully in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
files = [p for p in ROOT.rglob("*") if p.is_file()]
print(f"Found {len(files)} files under {ROOT}")

for p in tqdm(files, desc="auditing"):
    row = {
        "path": p.as_posix(),
        "cls": p.parent.name,          
        "file_size": p.stat().st_size,
        "sha256": sha256_of(p),
        "corrupt": False,
        "error": "",
    }
    try:
        with Image.open(p) as im:
            im.verify()                
        with Image.open(p) as im:      
            row.update(
                width=im.width,
                height=im.height,
                mode=im.mode,
                format=im.format,
            )
    except Exception as e:             
        row["corrupt"] = True
        row["error"] = str(e)[:100]
    rows.append(row)

df = pd.DataFrame(rows)

df["duplicate"] = df.duplicated("sha256", keep="first")
classes_per_hash = df.groupby("sha256")["cls"].nunique()
conflict_hashes = classes_per_hash[classes_per_hash > 1].index
df["cross_class_dup"] = df["sha256"].isin(conflict_hashes)

df.to_csv(OUT / "audit.csv", index=False)

ok = df[~df.corrupt]

print("\n== Images per class ==")
print(df.groupby("cls").size())

print("\n== width / height / file_size: min, median, max per class ==")
summary = ok.groupby("cls")[["width", "height", "file_size"]].agg(
    ["min", "median", "max"]
)
print(summary.T)

print("\n== Colour modes per class ==")
print(ok.groupby(["cls", "mode"]).size())

print("\n== Formats per class ==")
print(ok.groupby(["cls", "format"]).size())

print("\n== Problems ==")
print(f"corrupt / unreadable files : {int(df.corrupt.sum())}")
print(f"exact duplicates (extras)  : {int(df.duplicate.sum())}")
print(f"hashes present in BOTH classes: {len(conflict_hashes)}")

if df.corrupt.any():
    print("\nCorrupt files:")
    print(df.loc[df.corrupt, ["path", "error"]].to_string(index=False))

print(f"\nSaved table to {OUT / 'audit.csv'}")