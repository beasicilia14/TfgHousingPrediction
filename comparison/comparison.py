# IMPRIMIR CONTENIDO DEL ARCHIVO METRICS.CSV DE TODOS LOS MODELOS DIFERENTES 
import pandas as pd
import os

# Directorio externo
external_directory = os.path.abspath('.')

# Listar los directorios en el directorio externo
directories = os.listdir(external_directory)
print(directories)

# Crear un DataFrame vacío para almacenar las métricas
metrics_df = pd.DataFrame()

# Iterar a través de los directorios
for directory in directories:
    # Verificar si el directorio es un directorio y no un archivo
    if os.path.isdir(directory):
        # Obtener la ruta al archivo metrics.csv
        metrics_file = os.path.join(directory, 'metrics.csv')
        # Verificar si el archivo metrics.csv existe en el directorio
        if os.path.exists(metrics_file):
            # Leer el archivo metrics.csv en un DataFrame
            metrics = pd.read_csv(metrics_file)
            # Agregar una columna al DataFrame para almacenar el nombre del modelo
            metrics['Modelo'] = directory
            # Agregar las métricas al DataFrame metrics_df
            metrics_df = metrics_df.append(metrics)

# Restablecer el índice del DataFrame metrics_df
metrics_df = metrics_df.reset_index(drop=True)

# Imprimir el DataFrame metrics_df
print(metrics_df)

# Escribir en un archivo CSV llamado final_metrics.csv
metrics_df.to_csv("comparison//final_metrics.csv")

# Graficar la comparación de los modelos en un gráfico de barras, r2, mape y mae 
import matplotlib.pyplot as plt

# Obtener los nombres de modelo únicos
modelos = metrics_df['Modelo'].unique()

# Crear una figura y ejes
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Iterar a través de las métricas y graficar la comparación
for i, metrica in enumerate(['r2', 'mape', 'mae']):
    # Crear un gráfico de barras para la métrica
    ax[i].bar(modelos, metrics_df.groupby('Modelo')[metrica].mean())
    ax[i].set_title(metrica.upper())
    ax[i].set_ylabel(metrica.upper())
    ax[i].set_xlabel('Modelo')
    ax[i].set_xticklabels(modelos, rotation=45)

# Ajustar el diseño
plt.tight_layout()

plt.show()

plt.savefig('comparison//final_metrics.png')
