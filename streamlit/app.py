# Importamos las librerías necesarias.
import streamlit as st
import pandas as pd
import joblib  # Para cargar el modelo y el scaler.
import numpy as np

# Hacer la configuración de la página.
st.title("Predicción de Enfermedades Cardíacas")
st.write("Dime los valores y obtendrás una predicción sobre la probabilidad de enfermedad cardíaca.")

# Cargar el modelo y el StandardScaler desde los archivos locales.
modelo = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Crear un formulario para introducir los datos del paciente.
age = st.number_input("Edad", min_value=0, max_value=120, value=45)
trestbps = st.number_input("Presión arterial en reposo (mmHg)", min_value=80, max_value=200, value=130)
chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=400, value=210)
fbs = st.selectbox("Glucemia en ayunas > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resultado del electrocardiograma", [0, 1, 2])
thalach = st.number_input("Frecuencia cardíaca máxima alcanzada", min_value=60, max_value=220, value=150)
exang = st.selectbox("Angina inducida por el ejercicio", [0, 1])
oldpeak = st.number_input("Depresión del ST", min_value=0.0, max_value=10.0, value=1.0)
ca = st.number_input("Número de vasos coloreados por fluoroscopia", min_value=0, max_value=3, value=0)
sex_0 = st.selectbox("Sexo: Femenino (0) o Masculino (1)", [0, 1])

# Crear un DataFrame con los datos numéricos para normalizarlos.
numerical_data = pd.DataFrame({
    "age": [age],
    "trestbps": [trestbps],
    "chol": [chol],
    "thalach": [thalach],
    "oldpeak": [oldpeak]
})

numerical_data_scaled = pd.DataFrame(scaler.transform(numerical_data), columns=numerical_data.columns)

# Mostrar los datos originales y escalados.
st.write("### Datos originales:")
st.write(numerical_data)

st.write("### Datos escalados:")
st.write(numerical_data_scaled)

# Actualizar las variables normalizadas.
age_scaled, trestbps_scaled, chol_scaled, thalach_scaled, oldpeak_scaled = numerical_data_scaled.iloc[0]

# Crear el diccionario con los datos normalizados del paciente.
data = {
    "age": age_scaled,
    "trestbps": trestbps_scaled,
    "chol": chol_scaled,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach_scaled,
    "exang": exang,
    "oldpeak": oldpeak_scaled,
    "ca": ca,
    "sex_0.0": sex_0,
    "sex_1.0": 1 - sex_0,
    "cp_1.0": 1,
    "cp_2.0": 0,
    "cp_3.0": 0,
    "cp_4.0": 0,
    "thal_3.0": 1,
    "thal_6.0": 0,
    "thal_7.0": 0,
    "slope_1.0": 1,
    "slope_2.0": 0,
    "slope_3.0": 0
}

# Preparar los datos para la predicción.
input_data = pd.DataFrame([data])

if st.button("Predecir"):
    try:
        # Hacer la predicción con el modelo cargado.
        prediccion = modelo.predict(input_data)
        
        # Mostrar el resultado de la predicción.
        if prediccion[0] == 1:
            st.error("¡ALERTA! El modelo predice que el paciente tiene riesgo de enfermedad cardíaca.")
        else:
            st.success("¡A SALVO! El modelo predice que el paciente no tiene riesgo de enfermedad cardíaca.")
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
