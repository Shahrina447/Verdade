import pandas as pd

df = pd.read_csv("test.csv")

print("=== Label column mapping ===")
print(df.groupby(["Label", "label"]).size().to_string())

print("\n=== Sample label=0 rows ===")
for _, r in df[df["label"] == 0][["News Items", "Label", "label"]].head(3).iterrows():
    print(f"  label={r['label']} | Label={r['Label']} | text={str(r['News Items'])[:80]}")

print("\n=== Sample label=1 rows ===")
for _, r in df[df["label"] == 1][["News Items", "Label", "label"]].head(3).iterrows():
    print(f"  label={r['label']} | Label={r['Label']} | text={str(r['News Items'])[:80]}")

print("\n=== Sr. No. range ===")
print(f"Min: {df['Sr. No.'].min()}, Max: {df['Sr. No.'].max()}, Unique: {df['Sr. No.'].nunique()}")

print("\n=== Very short texts (< 30 chars) ===")
short = df[df["News Items"].astype(str).str.len() < 30]
print(f"Count: {len(short)}")
print(short[["News Items", "label"]].head(5).to_string())

# Check if test_400 has overlap with rest of test.csv
print("\n=== Checking if test_400 samples appear elsewhere in test.csv ===")
t400 = pd.read_csv("test_400.csv")
all_test_texts = set(df["News Items"].astype(str))
t400_texts = t400["News Items"].astype(str).tolist()
in_full = sum(1 for t in t400_texts if t in all_test_texts)
print(f"test_400 rows found in full test.csv: {in_full}/400 (expected 400)")

# Key question: is this truly a held-out test set?
# Check Sr. No. gaps suggesting train/test split
print("\n=== Sr. No. distribution (first 10 unique) ===")
print(sorted(df["Sr. No."].unique())[:10])
print(f"Total unique Sr. No.: {df['Sr. No.'].nunique()} / {len(df)} rows")
