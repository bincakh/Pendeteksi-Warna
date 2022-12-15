import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#Database
#X= Data, y = Target
#X = [[8],[10],[12],[14],[16],[18],[20],[22],[24],[26]]
#y = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0]
FileDB = 'perkalian.txt'
Database = pd.read_csv(FileDB, sep=",", header=0)
print ("---------------------")
print (Database)
#X = data, y = target
X = Database[[u'Feature']] #ciri1, ciri2, dst
y = Database.Target
regr = LinearRegression().fit(X, y)
regr.score(X, y)

#Data uji
predict = np.array([[98.0]])

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output ", regr.predict (predict))
