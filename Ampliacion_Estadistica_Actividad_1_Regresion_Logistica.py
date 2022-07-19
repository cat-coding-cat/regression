#Este programa es una secuencia de instrucciones, tal como se necesitan
# 1. Ordena el programa importando todo lo necesario en el encabezado
import pandas as pd

# tik_tok.csv contiene dos columnas
# age : edad de una persona
# has_tick_tok: 0 or 1 indicando si la persona tiene o no una cuenta de tik tok
df = pd.read_csv('./tik_tok.csv')
print(df)

from matplotlib import pyplot as plt

plt.scatter(df.age,df.has_tik_tok)
plt.show()

# visualizamos la forma de la data frame 
print(df.shape)

#  split the database int test and train

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(df[['age']],df.has_tik_tok)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

print( model.predict(x_test) )

score = model.score(x_test,y_test)

print(score)












