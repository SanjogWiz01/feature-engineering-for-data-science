# Ordinal Encoding Example

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Sample dataset
data = {
    "Size": ["Small", "Medium", "Large", "Medium"]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Define correct order
encoder = OrdinalEncoder(categories=[["Small", "Medium", "Large"]])

df["Size_encoded"] = encoder.fit_transform(df[["Size"]])

print("\nOrdinal Encoded Data:")
print(df)