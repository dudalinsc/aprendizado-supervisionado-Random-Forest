#atividade de aprendizado supervisionado
# Projeto: Classificação de filmes do IMDb usando aprendizado supervisionado
# Aluna: Maria Eduarda Lins Carrilho

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("imdb_top_1000.csv")

print("Dimensão da base:", df.shape)
print(df.info())
print(df.isnull().sum())

#Limpeza dos dados
#Remover colunas irrelevantes
df = df.drop(columns=["Poster_Link", "Overview", "Series_Title", "Director", 
                      "Star1", "Star2", "Star3", "Star4"])

#Remover linhas com dados faltantes
df = df.dropna()

#Remover símbolo $ e vírgula e converter 'Gross' para numérico
df["Gross"] = df["Gross"].replace('[\$,]', '', regex=True).astype(float)

print("Tamanho após limpeza:", df.shape)

#Criar variável alvo
#1 se IMDB_Rating >= 8.0, caso contrário 0
df["High_Rating"] = (df["IMDB_Rating"] >= 8.0).astype(int)

#Seleção de features
features = ["Released_Year", "Certificate", "Runtime", "Genre", "Meta_score", "Gross", "No_of_Votes"]
X = df[features].copy()  # <- cópia segura
y = df["High_Rating"]

#Pré-processamento
#Converter 'Runtime' (ex: '142 min') para número
X.loc[:, "Runtime"] = X["Runtime"].str.replace(" min", "", regex=False).astype(int)

#Codificar colunas categóricas
categorical_cols = ["Released_Year", "Certificate", "Genre"]
X_encoded = pd.get_dummies(X, columns=categorical_cols)

#Normalizar valores numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_processed = pd.DataFrame(X_scaled, columns=X_encoded.columns)

#Separação dos dados
X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}")

#Aprendizado Supervisionado escolhido:random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Avaliação
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Acurácia no teste:", acc)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

#Gráficos 
#Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

#Distribuição das notas de filmes
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="IMDB_Rating", kde=True, bins=20)
plt.title("Distribuição das Notas dos Filmes")
plt.xlabel("Nota IMDB")
plt.ylabel("Quantidade de Filmes")
plt.show()

#Importância das features
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(8, 4))
plt.barh(X_encoded.columns[indices], importances[indices])
plt.title("Top 10 Features Mais Importantes")
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.show()
