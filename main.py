import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import torch
from model import ModelMLP
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as prec
from sklearn.metrics import recall_score as rec
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt

#Читаем данные
df = pd.read_csv("churn_modeling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"]) #Очевидно ненужные признаки
df = pd.get_dummies(df, columns=['Geography', 'Gender'], dtype=int) #Превращаем эти колонки в one-hot кодирование
cols = df.columns

X = df.drop(columns=['Exited'])
y = df['Exited']
print(np.sum(np.array(y)))
#теперь мы имеем выборку без лишних признаков. Остался скейлинг

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
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

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(sum(y_train_balanced), len(y_train_balanced))

model = ModelMLP(X_train_scaled.shape[1])
model.fit(X_train=X_train_balanced,
          y_train=torch.from_numpy(y_train_balanced.values.copy()).float(),
          opt_name="Adam",
          epochs=10,
          batch_size=512,
          X_val=X_val_scaled,
          y_val=torch.from_numpy(y_val.values.copy()).float(),
          lr=0.01)

train_loss = model.loss_history
val_loss = model.val_loss_history
x = np.arange(0, len(val_loss))

plt.plot(train_loss, label='Train')
#plt.plot(val_loss, label='Validation')
plt.legend()
plt.show()
print(f"Train points: {len(train_loss)}")  # должно быть = эпохам
print(f"Val points: {len(val_loss)}")
print(model.loss_history)# должно быть = эпохам (если есть валидация)