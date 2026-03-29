
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN")

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop unnecessary columns
df = df.drop(["CustomerID", "Unnamed: 0"], axis=1, errors='ignore')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

upload_file("train.csv", "train.csv", "gayathri1909/tourism.csv", repo_type="dataset", token=HF_TOKEN)
upload_file("test.csv", "test.csv", "gayathri1909/tourism.csv", repo_type="dataset", token=HF_TOKEN)

print("Data prep completed")
