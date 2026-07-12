import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Create NumPy array
narr = np.array([
    [1, 2, 3, 4],
    [np.nan, 6, 7, 8],
    [9, 10, np.nan, 12]
])

print(narr)

# Create DataFrame
df = pd.DataFrame(narr, columns=['A', 'B', 'C', 'D'])

# Mean Imputation
print("------ Imputed Data with Mean ------")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_data = imputer.fit_transform(df)
print(imputed_data)

# Median Imputation
print("------ Imputed Data with Median ------")
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data = imputer.fit_transform(df)
print(imputed_data)

# Constant Imputation
print("------ Imputed Data with Constant ------")
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imputed_data = imputer.fit_transform(df)
print(imputed_data)
