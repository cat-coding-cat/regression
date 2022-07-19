#  LOGISTIC REGRESSION TAREA


#Este programa es una secuencia de instrucciones, tal como se necesitan
# 1. Ordena el programa importando todo lo necesario en el encabezado
import pandas as pd

# La siguiente base de datos

# https://raw.githubusercontent.com/sam16tyagi/Machine-Learning-techniques-in-python/master/logistic%20regression%20dataset-Social_Network_Ads.csv

# tienen variables
# User ID,Gender,Age,EstimatedSalary,Purchased


# 2. Cual es la variable que mejor classificación/prediccion ofrece para la veriable: Purchased

# Recuerda que la documentación de la función LogisticRegression() te puede ayudar https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


df = pd.read_csv('https://raw.githubusercontent.com/sam16tyagi/Machine-Learning-techniques-in-python/master/logistic%20regression%20dataset-Social_Network_Ads.csv')
print(df)

from matplotlib import pyplot as plt

plt.scatter(df.Age,df.Purchased)
plt.show()

# visualizamos la forma de la data frame 
print(df.shape)

#  split the database int test and train

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(df[['Age']],df.Purchased)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

print( model.predict(x_test) )

# 3. Imprime el score del modelo

score = model.score(x_test,y_test)

print(score)












