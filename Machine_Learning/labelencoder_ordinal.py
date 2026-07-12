from sklearn.preprocessing import LabelEncoder

# Ordinal data
ordinal_data = ['L', 'M', 'M', 'XL', 'XL']

# Categories
ordinal_categories = ['XL', 'L', 'M']

# Create LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder
label_encoder.fit(ordinal_categories)

# Transform the data
encoded_ordinal_data = label_encoder.transform(ordinal_data)

# Print results
print("Original ordinal data:", ordinal_data)
print("Encoded ordinal data:", encoded_ordinal_data)

# Decode back
decoded_data = label_encoder.inverse_transform(encoded_ordinal_data)
print("Decoded data:", decoded_data)
