import pandas as pd

# Load Excel file
df = pd.read_excel("evaluation/dataset/evaluation_dataset/Topographically selective.xlsx")

# Save as CSV
df.to_csv("evaluation/dataset/evaluation_dataset/topographically_selective.csv", index=False)
