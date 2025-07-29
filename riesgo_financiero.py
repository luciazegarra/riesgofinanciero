# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:35:10 2025

@author: zegar
"""
import pandas as pd
import seaborn as sns
import numpy as st 
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#Mostrar o cargar los datos

ds=pd.read_csv("dataset_financiero_riesgo.csv")


#Colocar un titulo principal en la pagina web
st.title("Predicción de riesgo financiero")

#Cargar los datos en la memoria cache para mejorar la velocidad del acceso al conjunto de datos
@st.cache_data
def cargar_datos():
    ds=pd.read_csv("dataset_financiero_riesgo.csv")
    return ds


# Recibiendo los datos carcagdos en la variable df (antes llamado dataset)
ds = cargar_datos()

# Mostrar los primeros datos (cino primeros datos)

st.write("Vista previa de los datos")
st.dataframe(ds.head())

#Preprocesamiento de datos o del conjunto de datos
#Cuando la variable es ordinal categorica, se debe usar esta formula

ds_encode=ds.copy() #Copia el dataset completo a otro dataset

label_cols=["Historial_Credito","Nivel_Educacion"]
le=LabelEncoder()
for col in label_cols: 
    ds_encode[col]=le.fit_transform(ds_encode[col])

x=ds_encode.drop("Riesgo_Financiero",axis=1)
y=ds_encode["Riesgo_Financiero"]
y=LabelEncoder().fit_transform(y)

#dividir en conjunto de entrenamiento y conjunto de test

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.8, random_state=0)


#Entrenar el modelo

modelo=RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(x_train, y_train)
score=modelo.score(x_test, y_test)

st.subheader(f"Precisión del modelo: {score:.2f}")

#Matriz de confusión

#nos ayuda a saber que datos nos sirven y que datos no nos sirve

y_pred=modelo.predict(x_test)
mc=confusion_matrix(y_test,y_pred)
st.subheader("Matriz de Confusión")
fig,ax=plt.subplots()

sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)










