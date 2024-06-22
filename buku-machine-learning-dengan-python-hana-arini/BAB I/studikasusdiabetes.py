import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


diabetes = pd.read_csv('diabetesIndo.csv') #maksudnya sep= adalah untuk memberi tahu pandas bahwa pemisah adalah titik koma
print(diabetes.columns)

# langkah 2 : menampilkan data
print("===============================================")

print( diabetes.head())
print("===============================================")

#langkah 3 memperhatikan dimensi data
print("===============================================")

print("dimensi data diabetes: {}".format(diabetes.shape))
print("===============================================")

# Langkah 4 untik menghitung tiap kelas
print("===============================================")

# identifikasi ada berapa diabetes , 1 = yes ; 0 = no
print(diabetes.groupby('Luaran').size())
print("===============================================")

# langkah 5 menampilkan data per faktor dengan berdasarkan klasifikasi luaran
print("===============================================")

faktor = diabetes.groupby('Luaran').hist(figsize=(9,9))
# faktor = diabetes.hist(figsize=(9,9))
# plt.show()
print(faktor)
print("===============================================")

print("===============================================")
# memastikan tidak ada missing data 

print(diabetes.isnull().sum())
print(diabetes.isna().sum())
print("===============================================")


print("===============================================")
# dari data diketahui jika terdapat sekitar 35 orang bertekanan darah 0 dimana tidak mungkin seseorang bertekanan darah 0
# proses ini akan menelusuri orang yang bertekanan darah 0

tekanan_darah_nol = diabetes[diabetes['TekananDarah'] == 0]
tekanan_darah_nol_jumlah = diabetes[diabetes['TekananDarah'] == 0].shape[0]
tekanan_darah_nol_grouped = tekanan_darah_nol.groupby('Luaran').size()
print(tekanan_darah_nol_grouped)
print("Total",tekanan_darah_nol_jumlah)
print("===============================================")


print("===============================================")
# level glukosa plasma , meskipun seseorang berpuasa tidak mungkin glukosa plasma menjadi 0 ini pasti ada kesalahan data

glukosa_plasma_nol = diabetes[diabetes['Glukosa']==0]
glukosa_plasma_nol_jumlah = diabetes[diabetes['Glukosa']==0].shape[0]
glukosa_plasma_nol_gruped = glukosa_plasma_nol.groupby('Luaran')['Umur'].count() # banyak cara , catatan baru
print(glukosa_plasma_nol_gruped)
print("Total",glukosa_plasma_nol_jumlah)
print("===============================================")

# ini step 10
print("===============================================")

kulit_10 = diabetes[diabetes['TebalKulit']==0]
kulit_jumlah = diabetes[diabetes['TebalKulit']==0].shape[0]
kulit_jumlah_grup = kulit_10.groupby('Luaran')['Umur'].count()
print(kulit_jumlah_grup)
print("Total",kulit_jumlah)
("===============================================")

# ini step 11
print("===============================================")

# bmi_k0 = diabetes[diabetes['BMI']==0]
# bmi_k0_jml = diabetes[diabetes['BMI']==0].shape[0]
# bmi_k0_jml_grup = kulit_10.groupby('Luaran')['Umur'].count()
# print(bmi_k0_jml_grup)
# print("Total",bmi_k0_jml)
print("Total :" , diabetes[diabetes.BMI == 0].shape[0])
print(diabetes[diabetes.BMI == 0 ].groupby('Luaran')['Umur'].count())
("===============================================")

# ini step 12
print("===============================================")

print("Total :" , diabetes[diabetes.Insulin == 0].shape[0])
print(diabetes[diabetes.Insulin == 0 ].groupby('Luaran')['Umur'].count())
("===============================================")

# ini step 13
("===============================================")
# dikarenakan kode2 diatas dianggap tidak valid maka akan dihapus

diabetes_mod = diabetes[(diabetes.TekananDarah != 0)&(diabetes.BMI !=0)&(diabetes.Glukosa !=0)]
print(diabetes_mod.shape)

nama_faktor = ['Kehamilan','Glukosa','TekananDarah','TebalKulit','Insulin','BMI','FungsiPedigree','Umur']
x = diabetes_mod[nama_faktor]
y = diabetes_mod.Luaran 
("===============================================")

# ini step 14
("===============================================")
models = []
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVC',SVC()))
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('GNB',GaussianNB()))
models.append(('RF',RandomForestClassifier()))
models.append(('GB',GradientBoostingClassifier()))
("===============================================")

# step 15
("===============================================")
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify = diabetes_mod.Luaran,random_state=0)
("===============================================")

# step 16
("===============================================")
names = []
scores = []

for name , model in models:
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Nama':names,'Nilai':scores})
print(tr_split)
("===============================================")

# langkah 17
("===============================================")
names = []
scores=[]
for name , model in models:
    kfold = KFold(n_splits = 10,random_state = 10)
    score = cross_val_score(model,x,y,cv=kfold,scoring='accuracy'),mean()
    names.append(name)
    score.append(score)
kf_cross_val = pd.DataFrame({'Nama':names,'Nilai':score})
print(kf_cross_val)
("===============================================")
