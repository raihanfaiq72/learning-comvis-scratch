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
# plt.show()
print(faktor)
print("===============================================")

print("===============================================")
# memastikan tidak ada missing data 
print("langkah 6 memastikan tidak ada missing data")
print(diabetes.isnull().sum())
print(diabetes.isna().sum())
print("===============================================")


print("===============================================")
# dari data diketahui jika terdapat sekitar 35 orang bertekanan darah 0 dimana tidak mungkin seseorang bertekanan darah 0
# proses ini akan menelusuri orang yang bertekanan darah 0
print("langkah 7 , Menelusuri orang dengan tekanan darah 0")
tekanan_darah_nol = diabetes[diabetes['TekananDarah'] == 0]
tekanan_darah_nol_jumlah = diabetes[diabetes['TekananDarah'] == 0].shape[0]
tekanan_darah_nol_grouped = tekanan_darah_nol.groupby('Luaran').size()
print(tekanan_darah_nol_grouped)
print("Total",tekanan_darah_nol_jumlah)
print("===============================================")


print("===============================================")
# level glukosa plasma , meskipun seseorang berpuasa tidak mungkin glukosa plasma menjadi 0 ini pasti ada kesalahan data
print("Langkah 8 , orang dengan glukosa plasma 0")
glukosa_plasma_nol = diabetes[diabetes['Glukosa']==0]
glukosa_plasma_nol_jumlah = diabetes[diabetes['Glukosa']==0].shape[0]
glukosa_plasma_nol_gruped = glukosa_plasma_nol.groupby('Luaran')['Umur'].count() # banyak cara , catatan baru
print(glukosa_plasma_nol_gruped)
print("Total",glukosa_plasma_nol_jumlah)
print("===============================================")