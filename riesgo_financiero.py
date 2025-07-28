# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:35:10 2025

@author: zegar
"""
import pandas as pd
import seaborn as sb
import numpy as st 
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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









