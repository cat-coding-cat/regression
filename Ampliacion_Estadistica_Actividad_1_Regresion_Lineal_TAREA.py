#----------------------------------------------------------------------------------------
# ESTAS SON LAS LIBRERIAS QUE UTILIZAMOS
# matplotlib es para hacer graficos
# numpy permite manejar arreglos ( arrays)
# sklearn contiene herramientas para Machine Learning: Algoritmos, metricas para validar
# sklearn también contiene datasets
#----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Cargamos el dataset diabetes. Tenemos un vector X con distintas variables que medimos de una # persona, y la variable y que nos diría "el grado de diabetes" que tiene
# Recordemos que la diabetes no es una enfermedad binaria sino que se tiene "en cierto grado"

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


#----------------------------------------------------------------------------------------
#visualizamos X,y para darinos una idea del dataset
print(diabetes_X)
print(diabetes_y)

#### Puedo seleccionar sólo un feature (una veriable del vector X)

#diabetes_X = diabetes_X[:, np.newaxis, 0] # Cojo la 0
#diabetes_X = diabetes_X[:, np.newaxis, 4]  # Cojo la 4
#
print(diabetes_X)

print(len(diabetes_X))


#----------------------------------------------------------------------------------------
#  TAREA 
## 1. Probar distintos features (variables) y ver cual es la que produce los mejores resultados.
## 2. Utilizar el conjunto completo de features e imprimir el resultado 
#     (recuerda que para ello debes desactivar la grafica solo estamos graficando abajo 2 dimensiones)


#----------------------------------------------------------------------------------------
# PARA ENTRENAR EL MODELO Y TENER CON QUE PROBARLO LO SEPARAMOS EN train y test

### Split the data into training/testing sets
# Cojo 20 datos
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Cojo 20 datos
### Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
#----------------------------------------------------------------------------------------





#----------------------------------------------------------------------------------------
#    ENTRENAMOS NUESTRO MODELO DE MACHINE LEARNING

##  1. Creamos el modelo: the linear regression object
regression = linear_model.LinearRegression()


### 2. Entrenamos (.fit) el modelo usando los training sets
regression.fit(diabetes_X_train, diabetes_y_train)

### 3. Hacemos predicciones y evaluamos el modelo usando el testing set
diabetes_y_pred = regression.predict(diabetes_X_test)


#----------------------------------------------------------------------------------------
# IMPRIMIMOS LOS RESULTADOS!!
### The coefficients
# si utilizamos una sola variable tendremos 1 solo coeficiente
# si utilizamos todo el dataset tendremos 10
print( "Coefficients: \n", regression.coef_ )

# Termino independiente en el modelo lineal
print( "Intercept: \n", regression.intercept_ )

### The mean squared error
print( "Mean squared error: %.2f" % mean_squared_error( diabetes_y_test, diabetes_y_pred ) )

#
### Plot el modelo (la recta) y los datos originales
#plt.ylabel('y')
#plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
#plt.ylabel('y')
#plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
#plt.xlabel('x')
#plt.show()
