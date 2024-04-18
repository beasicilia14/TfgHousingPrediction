# TFG Housing Prediction

AUTOR: Beatriz Sicilia Gómez 
GRADO: 5ºGITT + BA

Este proyecto se centra en el desarrollo de modelos de Machine Learning para predecir el precio de venta de viviendas en Madrid. Utilizando datos obtenidos a través de la API del portal inmobiliario Idealista, se ha almacenado la información en la nube mediante Firebase. Posteriormente, se llevó a cabo un exhaustivo análisis exploratorio y limpieza de los datos.

Los modelos se han construido haciendo uso de las bibliotecas SciKit-Learn, XGBoost y TensorFlow-Keras.

## Estructura del Proyecto

### Carpeta dataCollection:

Aquí se encuentran dos funciones clave:

- **llamada_api.py**: Esta función permite llamar a la API de Idealista utilizando las claves obtenidas y los parámetros deseados, almacenando los resultados en archivos .json.

- **loading_data.py**: Utilizada para cargar los datos en una colección inicial de Firestore Firebase.

### Carpeta dataPreprocessing&EDA:

- **dataCleansing.py**: Carga los datos en crudo desde Firebase y genera una nueva colección eliminando los valores nulos (NAs), realizando selección de variables y creando nuevas características (feature engineering).

- **EDA.ipynb**: Contiene código con visualizaciones para el análisis exploratorio de los datos.

### Carpeta para Cada Modelo:

Cada modelo tiene la siguiente estructura:

- **modelo.py**: Contiene el código del modelo, incluyendo las fases de entrenamiento, validación y prueba.

- **infomodelo.txt**: Archivo de texto donde se almacenan características del modelo como hiperparámetros y coeficientes.

- **metrics.csv**: Archivo que recopila las métricas obtenidas para el conjunto de prueba.

- **visualizaciones**: Incluye visualizaciones de los resultados del modelo.

### Carpeta comparison:

- **comparison.py**: Contiene el código que lee los archivos de métricas de todos los modelos, los une en un único dataframe y genera visualizaciones comparativas.

- **final_metrics.csv**: Archivo que recopila todas las métricas de los modelos.

- **final_metrics.png**: Imagen comparativa de las métricas.


### Anexo:
Nota: Se han excluido del repositorio los archivos con las claves de la API de Idealista y las credenciales de acceso a Firebase para garantizar la seguridad de los datos.

Dependencias: En el archivo requirements.txt se encuentran todas las bibliotecas y versiones necesarias para ejecutar este proyecto de manera adecuada. Para instalarlo, ejecutar el comando:
```bash
pip install -r requirements.txt
```

