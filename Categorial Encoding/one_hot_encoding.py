import pandas as pd

data = {
    "City":["Pokhara","Kathmandu","Butwal","Pokhara"]
}

df = pd.DataFrame(data)

encoded = pd.get_dummies(df["City"])

print(encoded)
# one hot = converting it into binary form