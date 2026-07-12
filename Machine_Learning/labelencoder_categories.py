from sklearn.preprocessing import LabelEncoder

categories =['red','blue','green','yellow','red','blue','green']
label_encoder=LabelEncoder()

encoded_data = label_encoder.fit_transform(categories)

print("Original categories:",categories)
print("Encoded data:", encoded_data)

decoded_data = label_encoder.inverse_transform(encoded_data)
print("Decoded data:",decoded_data)
