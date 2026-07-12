import pandas as pd
import numpy as np

df=pd.read_csv(r'D:/Programming Development/Machine_Learning/DataSet/tempbook.csv')
print(df)

size_mapping={'XL':3,'L':2,'M':1}
print(df)

inv_size_mapping={v:k for k,v in size_mapping.items()}
df['size']=df['size'].map(inv_size_mapping)
print(df)

class_mapping = {label:idx for idx, label in enumerate(np.unique(df['class']))}
print(class_mapping)
df['class']=df['class'].map(class_mapping)
print(df)

inv_class_mapping = {v:k for k ,v in class_mapping.items()}
df['class'] =df['class'].map(inv_class_mapping)
print(df)

from sklearn.preprocessing import LabelEncoder
class_le =LabelEncoder()
y=class_le.fit_transform(df['class'].values)
print(y)

class_le.inverse_transform(y)
print(y)
