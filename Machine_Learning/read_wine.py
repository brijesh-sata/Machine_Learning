import pandas as pd

# Read CSV file
winedatac = pd.read_csv(r'D:/Programming Development/Machine_Learning/DataSet/wine.csv')

# Display the DataFrame
print(winedatac)

# Display first 5 rows
print(winedatac.head())

# Display shape
print("Shape:\n", winedatac.shape)

# Display column names
print("Columns:\n", winedatac.columns)

# Display data types
print("Data Types:\n", winedatac.dtypes)

# Display dimensions
print("Number of Dimensions:\n", winedatac.ndim)

# Display total number of elements
print("Size:\n", winedatac.size)
