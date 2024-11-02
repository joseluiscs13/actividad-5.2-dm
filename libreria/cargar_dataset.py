import pandas as pd

ruta_archivo = 'data/dataset.csv'

class CargadorDatos:
    @staticmethod
    def cargar_archivo(ruta_archivo):
        #Carga el dataset y se convierte a dataframe
        try:
            return pd.read_csv(ruta_archivo)
        except Exception as e:
            raise ValueError(f"No se pudo cargar el archivo: {e}")
