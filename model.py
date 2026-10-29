import glob, os
import pandas as pd
import numpy as np

files = sorted(glob.glob(os.path.join("CIC-IOT-2023/","*.csv")))
# print(len(files), "files")
# print(files[:5])

df = pd.read_csv(files[0])
print("rows, columns:", df.shape)
df.head()
df.info()

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

