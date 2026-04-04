import pandas as pd
import numpy as np
import torch

df = pd.read_csv("churn_modeling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"]) #Очевидно ненужные признаки
print(df.head())
df = pd.get_dummies(df, columns=['Geography', 'Gender'], dtype=int) #Превращаем эти колонки в one-hot кодирование
cols = df.columns

X = df.drop(columns=['Exited']).values
y = df['Exited'].values
#теперь мы имеем преобразованные признаки.
print(X[0])



