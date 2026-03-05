# Label Encoding Example

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = {
    "City": ["Pokhara", "Kathmandu", "Butwal", "Pokhara"]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Apply Label Encoding
encoder = LabelEncoder()
df["City_encoded"] = encoder.fit_transform(df["City"])

print("\nEncoded Data:")
print(df)

# Show mapping
print("\nMapping of labels:")
for i, label in enumerate(encoder.classes_):
    print(label, "->", i)