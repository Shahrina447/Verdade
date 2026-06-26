import pandas as pd

df = pd.read_csv("test.csv")

print("=== REAL samples (label=0, Label=FAKE) ===")
for _, r in df[df["label"] == 0].head(15).iterrows():
    print(f"  [{r['label']}] {r['News Items'][:120]}")

print()
print("=== FAKE samples (label=1, Label=TRUE) ===")
for _, r in df[df["label"] == 1].head(15).iterrows():
    print(f"  [{r['label']}] {r['News Items'][:120]}")

print()
print("=== Text length stats per class ===")
df["len"] = df["News Items"].astype(str).str.len()
print(df.groupby("label")["len"].describe().to_string())
