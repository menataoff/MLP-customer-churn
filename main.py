import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import torch

#Читаем данные
df = pd.read_csv("churn_modeling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"]) #Очевидно ненужные признаки
df = pd.get_dummies(df, columns=['Geography', 'Gender'], dtype=int) #Превращаем эти колонки в one-hot кодирование
cols = df.columns

X = df.drop(columns=['Exited'])
y = df['Exited']
#теперь мы имеем выборку без лишних признаков. Остался скейлинг

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#0.64 train, 0.16 val, 0.2 test

#на самом деле понятно, что нужно скейлить, но посмотрим для интереса и убедимся на значения первых трех объектов
for idx, col in enumerate(cols.drop("Exited")):
    print(f'{col}: {X.iloc[0, idx]}, {X.iloc[1, idx]}, {X.iloc[2, idx]}')
#теперь окончательно убедились, что скейлим
features_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

ct = ColumnTransformer([
    ('scaler', StandardScaler(), features_to_scale)
], remainder='passthrough')

ct.fit(X_train)
X_train_scaled = torch.Tensor(ct.transform(X_train))
X_val_scaled = torch.Tensor(ct.transform(X_val))
X_test_scaled = torch.Tensor(ct.transform(X_test))

print(X_train_scaled[0])

model =



