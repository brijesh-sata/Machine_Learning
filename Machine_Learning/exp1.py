import numpy as np
import pandas as pd

# Create NumPy array with missing values
arr = np.array([
    [1, 2, 3, 4],
    [np.nan, 6, 7, 8],
    [9, 10, np.nan, 12]
])

# Create DataFrame
df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])

# Display DataFrame
print(df)

print("----1----")
# Count missing values
print(df.isnull().sum())

print("----2----")
# Display NumPy array
print(df.values)

print("----3----")
# Drop rows with any NaN values
print(df.dropna())

print("----4----")
# Drop columns with any NaN values
print(df.dropna(axis=1))

print("----5----")
# Drop rows where all values are NaN
print(df.dropna(how='all'))

print("----6----")
# Keep rows having at least 4 non-NaN values
print(df.dropna(thresh=4))

print("----7----")
# Drop rows where column A has NaN
print(df.dropna(subset=['A']))

print("----8----")
# Drop columns where all values are NaN
print(df.dropna(axis=1, how='all'))
