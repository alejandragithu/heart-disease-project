# Importar las librerías necesarias.
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

# Cargar el modelo.
model = joblib.load('heart_disease_model.pkl')

# Mostrar las columnas esperadas por el modelo.
print("Columnas esperadas por el modelo:", model.feature_names_in_)

# Crear la aplicación Flask.
app = Flask(__name__)

# Definir la ruta /predict y la función predict.
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Leer datos JSON.
    try:
        # Crear DataFrame a partir de los datos JSON.
        df = pd.DataFrame([data])
        
        # Asegurar que todas las columnas esperadas estén presentes.
        expected_columns = model.feature_names_in_
        df = df.reindex(columns=expected_columns, fill_value=0)
        
        # Realizar la predicción.
        prediction = model.predict(df)
        
        # Mostrar la predicción.
        print({'prediction': prediction.tolist()})

        # Devolver la predicción como respuesta JSON.
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Iniciar la aplicación Flask.
if __name__ == '__main__':
    app.run(debug=True)
