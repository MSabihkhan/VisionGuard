"""Clean the audit table and make stratified train/val/test lists."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("reports/audit.csv")

df = df[~df.corrupt]                       
df = df[~df.cross_class_dup]               
df = df.drop_duplicates("sha256", keep="first")   

df["label"] = (df.cls == df.cls.unique()[0]).astype(int)   
print(df.groupby(["cls", "label"]).size())

train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)

Path("splits").mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "cls", "label", "sha256"]].to_csv(f"splits/{name}.csv", index=False)
    print(name, len(part), part.label.mean().round(3))