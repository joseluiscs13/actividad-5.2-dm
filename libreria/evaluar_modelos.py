from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Evaluacion:
    @staticmethod
    def dividir_datos(X, y, test_size=0.3):
        #Divide los datos en entrenamiento y prueba
        return train_test_split(X, y, test_size=test_size, random_state=None)
    
    @staticmethod
    def evaluar_modelo(modelo, X, y, iteraciones=5, test_size=0.3):
       #Evalúa el modelo en múltiples iteraciones
       resultados = []
       for i in range(iteraciones):
           X_train, X_test, y_train, y_test = Evaluacion.dividir_datos(X, y, test_size)
           modelo.entrenar(X_train, y_train)
           y_pred = modelo.predecir(X_test)
           precision = accuracy_score (y_test, y_pred)
           resultados.append(precision)
           print(f"Iteración {i+1}, Precisión: {precision}")
       return resultados 

        

        
     
