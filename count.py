"""
count.py
--------
Shows exact counts and default breakdown for home ownership groups.

Run:
    python count.py
"""

import pandas as pd

df = pd.read_csv("cleaned_loanstats.csv", low_memory=False)

home_labels = {4: "OWN", 3: "MORTGAGE", 2: "RENT", 1: "OTHER"}
df["home_label"] = df["home_ownership"].map(home_labels)

print("=" * 65)
print("  Home Ownership — Count and Default Breakdown")
print("=" * 65)

total = len(df)

for label in ["OWN", "MORTGAGE", "RENT", "OTHER"]:
    group      = df[df["home_label"] == label]
    count      = len(group)
    defaulted  = group["default"].sum()
    paid_off   = count - defaulted
    def_rate   = defaulted / count * 100
    share      = count / total * 100

    print(f"\n  {label}")
    print(f"    Total borrowers  : {count:>10,}  ({share:.1f}% of all borrowers)")
    print(f"    Paid off         : {paid_off:>10,}  ({100 - def_rate:.1f}%)")
    print(f"    Defaulted        : {defaulted:>10,}  ({def_rate:.1f}%)")

print("\n" + "=" * 65)
print(f"  Total borrowers  : {total:>10,}")
print(f"  Total defaulted  : {df['default'].sum():>10,}  ({df['default'].mean()*100:.1f}%)")
print("=" * 65)
