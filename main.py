import pandas as pd
import numpy as np
import torch

df = pd.read_csv("churn_modeling.csv")
print(df.head())
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
cols = df.columns
print(cols)
X = np.array(df)[:, :-1]
y = np.array(df)[:, -1]

for x, col in zip(X[0], cols[:-1]):
    print(f'Значение колонки {col}: {x}')

#Видно, что 0, 3, 5, 9 - точно числовые колонки
#Видно, что 1, 2, 7, 8 - точно категориальные колонки
#Посмотрим, что с остальными: 4, 6

idxs_to_check = [4,6]

for idx in idxs_to_check:
    print(f"Множество значений колонки {cols[idx]}: {set(X[:, idx])}")

#Видно что обе колонки можно считать категориальными,
#хотя по факту они числовые (срок владения и число продуктов)
#Хотя, наверное я путаю понятия категориальных и не категориальных признаков.
#Так то везде кроме колонок 1 и 2 признаки числовые и с ними надо работать просто как с числом.
#Зато потренировался с индексацией и срезами...



# data = torch.from_numpy(np.array(df))
# print(data[0])