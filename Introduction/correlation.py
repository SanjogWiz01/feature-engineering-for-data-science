import pandas as pd
data = {
    "Age":[22,25,30,35],
    "Salary":[20000,30000,40000,50000],
    "Bought":[0,1,1,0]
}

df = pd.DataFrame(data)

print(df.corr())
'''df.head()
df.info()
df.describe()
df.isnull().sum()
df.drop()
df.corr() '''