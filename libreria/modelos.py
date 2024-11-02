from sklearn.dummy import DummyClassifier
import pandas as pd

class ModeloZeroR:
    def __init__(self):
        self.modelo = DummyClassifier(strategy='most_frequent')
    
    def entrenar(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)

    def predecir(self, X_test):
        return self.modelo.predict(X_test)

class ModeloOneR:
    def __init__(self):
        self.mejor_atributo = None
        self.mejor_regla = None
        self.mejor_precision = 0.0

    def entrenar(self, X_train, y_train):
        self.mejor_atributo = None
        self.mejor_regla = None
        self.mejor_precision = 0.0

        #Iterar sobre cada atributo en X_train
        for columna in X_train.columns:
            #crear tabla de contigencia
            tabla_contingencia = pd.crosstab (X_train[columna], y_train)
            #obtener la clase más frecuente para cada valor del atributo
            predicciones = tabla_contingencia.idxmax(axis=1)
            #calcular las predicciones
            y_pred = X_train[columna].map(predicciones)
            #calcular precisión
            precision = (y_pred == y_train).mean()#precisión general
            if precision > self.mejor_precision:
                self.mejor_precision = precision
                self.mejor_atributo = columna
                self.mejor_regla = predicciones
        
    def predecir(self, X_test):
        if self.mejor_atributo is None:
            raise Exception("El modelo no ha sido entrenado. Llama a 'entrenar' primero. Paro")
        #hacer predicciones usando el mejor atributo
        return X_test[self.mejor_atributo].map(self.mejor_regla)


