import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv('diabetesIndo.csv') #maksudnya sep= adalah untuk memberi tahu pandas bahwa pemisah adalah titik koma
print(diabetes.columns)

# langkah 2 : menampilkan data
print("===============================================")
print("Ini adalah Langkah 2 , yaitu menampilkan data ")
print( diabetes.head())
print("===============================================")

#langkah 3 memperhatikan dimensi data
print("===============================================")
print("ini adalah langkah 3 , menampilkan dimensi data")
print("dimensi data diabetes: {}".format(diabetes.shape))
print("===============================================")

# Langkah 4 untik menghitung tiap kelas
print("===============================================")
print("ini adalah langkah 4 , menghitung tiap kelas")
# identifikasi ada berapa diabetes , 1 = yes ; 0 = no
print(diabetes.groupby('Luaran').size())
print("===============================================")

# langkah 5 menampilkan data per faktor dengan berdasarkan klasifikasi luaran
print("===============================================")
print("ini adalah langkah 5 , Menampilkan data perfaktor")
faktor = diabetes.groupby('Luaran').hist(figsize=(9,9))
# faktor = diabetes.hist(figsize=(9,9))
print(faktor)
print("===============================================")