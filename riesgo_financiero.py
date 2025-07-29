# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:35:10 2025

@author: zegar
"""

import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Mostrar o cargar los datos
st.title("Predicción de Riesgo Financiero")

# Cargar los datos en la memoria caché para mejorar la velocidad del acceso al conjunto de datos
@st.cache_data
def cargar_datos():
    ds = pd.read_csv("dataset_financiero_riesgo.csv")
    return ds

# Cargar dataset
ds = cargar_datos()

# Mostrar los primeros datos
st.write("Vista previa de los datos")
st.dataframe(ds.head())

# Preprocesamiento de datos
ds_encode = ds.copy()

# Codificar variables categóricas ordinales
label_cols = ["Historial_Credito", "Nivel_Educacion"]
le = LabelEncoder()
for col in label_cols:
    ds_encode[col] = le.fit_transform(ds_encode[col])

# Separar variables predictoras y variable objetivo
x = ds_encode.drop("Riesgo_Financiero", axis=1)
y = ds_encode["Riesgo_Financiero"]
y = LabelEncoder().fit_transform(y)

# Dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(x_train, y_train)

# Evaluar precisión
score = modelo.score(x_test, y_test)
st.subheader(f"Precisión del modelo: {score:.2f}")

# Matriz de confusión
y_pred = modelo.predict(x_test)
mc = confusion_matrix(y_test, y_pred)

st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Importancia de las características
importancias = modelo.feature_importances_
st.subheader("Importancia de las características")

importancia_ds = pd.DataFrame({
    "Caracteristicas": x.columns,
    "Importancia": importancias
})

st.bar_chart(importancia_ds.set_index("Caracteristicas"))






