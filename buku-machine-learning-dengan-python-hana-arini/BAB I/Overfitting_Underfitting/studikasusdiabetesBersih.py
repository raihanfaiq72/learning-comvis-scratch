import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv('diabetesIndo.csv')
diabetes.columns

diabetes_mod = diabetes[(diabetes.TekananDarah != 0) & (diabetes.BMI !=0) & (diabetes.Glukosa !=0)]
print(diabetes_mod.shape)

# simpan file csv bersihnya
diabetes_mod.to_csv('cleanDiabetes.csv')