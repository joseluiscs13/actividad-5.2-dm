import pandas as pd
from libreria.cargar_dataset import CargadorDatos
from libreria.modelos import ModeloZeroR,ModeloOneR
from libreria.evaluar_modelos import Evaluacion
from sklearn.model_selection import train_test_split

#Cargar el dataset
ruta_archivo = 'data/dataset.csv'
df = CargadorDatos.cargar_archivo(ruta_archivo)
#definir las características y la clase
X = df.drop(columns=['brand'])#clase
y = df['brand']

#Modelo ZeroR
modelo_zeroR = ModeloZeroR()
resultados_zeroR = Evaluacion.evaluar_modelo(modelo_zeroR, X, y, iteraciones=5)

#Modelo OneR
modelo_oneR = ModeloOneR()
resultados_oneR = Evaluacion.evaluar_modelo(modelo_oneR, X, y, iteraciones=5)

print("Resultados de ZeroR:",resultados_zeroR)
print("Resultados de OneR:",resultados_oneR)

#Valor más frecuente ZeroR
masfrecuente = y.value_counts().idxmax()
frecuencia = y.value_counts().max()
print(f"ZeroR, valor más frecuente: {masfrecuente}, Frecuencia: {frecuencia}")

#OneR mejor regla
print(f"Mejor atributo OneR: {modelo_oneR.mejor_atributo}")
print(f"Mejor regla OneR: {modelo_oneR.mejor_regla}")

#Errores OneR
X_train, X_test, y_train, y_test = Evaluacion.dividir_datos(X, y, test_size=0.3) #conjunto de prueba
modelo_oneR.entrenar(X_train, y_train)
y_pred = modelo_oneR.predecir(X_test)
erroresOneR = (y_pred != y_test).sum()
print(f"Resultado OneR, errores: {erroresOneR}")

