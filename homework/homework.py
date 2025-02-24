#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
import os
from glob import glob
from sklearn.linear_model import LinearRegression
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_data(ptr, ptt):
    train_dfp = pd.read_csv(ptr, index_col=False, compression="zip")
    test_dfp = pd.read_csv(ptt, index_col=False, compression="zip")
    return train_dfp, test_dfp

def preprocess_data(train_dfp, test_dfp):
    reference_year = 2021
    
    tctn = train_dfp.copy()
    ctt = test_dfp.copy()
    
    for df in [tctn, ctt]:
        df["Age"] = reference_year - df["Year"]
        df.drop(columns=['Year', 'Car_Name'], inplace=True)
    
    return tctn.dropna(), ctt.dropna()

def create_model_pipeline(categorical_features, numerical_features):
    """Create preprocessing and model pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), categorical_features),
            ('scaling', MinMaxScaler(), numerical_features),
        ]
    )
    
    return Pipeline([
        ("data_preprocessor", preprocessor),
        ('feature_selector', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])

def calculate_metrics(y_true, y_pred, dataset_type):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(median_absolute_error(y_true, y_pred))
    }

def main():
    # Load and prepare data
    train_dfp, test_dfp = load_data(
        "./files/input/train_data.csv.zip",
        "./files/input/test_data.csv.zip"
    )
    
    tctn, ctt = preprocess_data(train_dfp, test_dfp)
    
    # Split features and target
    x_train = tctn.drop(columns=["Present_Price"])
    y_train = tctn["Present_Price"]
    x_test = ctt.drop(columns=["Present_Price"])
    y_test = ctt["Present_Price"]
    
    ccat = ['Fuel_Type', 'Selling_type', 'Transmission']
    cnum = [col for col in x_train.columns if col not in ccat]
    
    # Create and configure pipeline
    pipeline = create_model_pipeline(ccat, cnum)
    
   
    gp = {
        'feature_selector__k': range(1, 25),
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False]
    }
    
    # Perform grid search
    model = GridSearchCV(
        estimator=pipeline,
        gp=gp,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    
    model.fit(x_train, y_train)
    
    # Save model
    if os.path.exists("files/models/"):
        for file in glob(f"files/models/*"):
            os.remove(file)
        os.rmdir("files/models/")
    os.makedirs("files/models/")
    
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)
    
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
   
    train_metrics = calculate_metrics(y_train, y_train_pred, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, "test")
    
    # Save results
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")

if __name__ == "__main__":
    main()