import pandas as pd
import numpy as np
import torch

df = pd.read_csv("churn_modeling.csv")
print(df.head())
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
cols = df.columns
df = np.array(df)

for col in cols:




# data = torch.from_numpy(np.array(df))
# print(data[0])