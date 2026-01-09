import matplotlib as plot
import pandas as pd
import csv 
import pickle as pk
import os

print("Hello World")


# Read a local CSV file
df = pd.read_csv('DBs/database.csv')

# You can also read a CSV file directly from a URL
# df = pd.read_csv(url)

# Display the first 5 rows of the DataFrame to verify
print(df.head())

