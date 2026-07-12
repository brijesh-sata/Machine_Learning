#Onehotencoder for mapping

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df=pd.read_csv(r"D:/Programming Development/Machine_Learning/DataSet/tempbook.csv")

print(df)

x=df[['color','size','price']].values
color_le = LabelEncoder()
x[:,0] = color_le.fit_transform(x[:,0])
print(x)

#one way with onehotencoder for creating dummy features whith sklearn
ct=ColumnTransformer([("color",OneHotEncoder(),[0])],remainder='passthrough')
x=ct.fit_transform(x)
print(x)

#Other way with pandas
print(pd.get_dummies(df[['color','price']]))
