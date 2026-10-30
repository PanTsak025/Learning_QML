import glob, os
import pandas as pd
import numpy as np

files = sorted(glob.glob(os.path.join("CIC-IOT-2023/","*.csv")))
# print(len(files), "files")
# print(files[:5])

df = pd.read_csv(files[0])
print("rows, columns:", df.shape)
# df.head()
# df.info()

label_col = df.columns[-1] #last column containing the labels
counts = df[label_col].value_counts()
print(counts.size, "labels in this file")
print(counts)

pd.set_option("display.max_columns", None)
df.describe().T

num = df.select_dtypes("number")
print("missing values: ", int(df.isna().sum().sum()))
print("infinite values: ", int(np.isinf(num).sum().sum()))
print("duplicate rows: ", int(df.duplicated().sum()))
print("constant columns: ", [c for c in num.columns if num[c].nunique() == 1])

bad = num.isna() | np.isinf(num)
print(bad.sum()[bad.sum()>0])   # columns affected by the nan/inf values
print(df.loc[bad.any(axis=1), label_col].value_counts()) # labels affected

dup_pct = df.duplicated().groupby(df[label_col]).mean() * 100
dup_pct.sort_values(ascending=False).round(1)

feat = df.columns.drop(label_col)
dedup = df.drop_duplicates()
print("feature rows with conflicting labels:", int(dedup.duplicated(subset=feat).sum()))

info = pd.DataFrame({
    "unique": num.nunique(),
    "min": num.min(),
    "max": num.max(),
    "only_0_1": num.isin([0,1]).all(),
})
print(info)

def family(label):
    s = label.upper()
    if s == "BENIGN":                                   return "Benign"
    if s.startswith("DDOS"):                            return "DDoS"
    if s.startswith("DOS"):                             return "DoS"
    if s.startswith("MIRAI"):                           return "Mirai"
    if s.startswith("RECON")or s =="VULNERABILITYSCAN": return "Recon"
    if "SPOOFING" in s:                                 return "Spoofing"
    if s == "DICTIONARYBRUTEFORCE":                     return "BruteForce"
    return "Web"    # BrowserHijacking, SQLi, CommandInjection, XSS, Backdoor, Uploading

df["family"] = df[label_col].map(family)
df["family"].value_counts()

print("Tot size == AVG:", (df["Tot size"] == df["AVG"]).mean())
print("corr(Variance, Std^2):", df["Variance"].corr(df["Std"] ** 2))

corr = num.replace([np.inf, -np.inf], np.nan).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
upper[upper > 0.95].sort_values(ascending=False)