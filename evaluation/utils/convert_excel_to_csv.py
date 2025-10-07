import pandas as pd

# Load Excel file
df = pd.read_excel("/home/gsasikiran/tib/code/llm-hackathon-MS-and-Chem-2025/evaluation/dataset/evaluation_dataset/Topographically selective.xlsx")

# Save as CSV
df.to_csv("/home/gsasikiran/tib/code/llm-hackathon-MS-and-Chem-2025/evaluation/dataset/evaluation_dataset/topographically_selective.csv", index=False)
