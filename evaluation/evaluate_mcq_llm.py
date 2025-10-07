import os
from openai import OpenAI
import csv
import json
from tqdm import tqdm


# Specify the folder containing CSV files
DATASET_FOLDER_PATH = "evaluation/dataset/evaluation_dataset"
# Open LLM Client
client = OpenAI()
MODEL_NAME = "o3" # other models: o4-mini, gpt-5

SYSTEM_PROMPT = """You will be provided with a question and four options. You have to provide correct answer as a single word string 'Option A' or 'Option B', 'Option C' or 'Option D'."""

# Load dataset

results = []

# Iterate over all files in the folder
for filename in os.listdir(DATASET_FOLDER_PATH):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATASET_FOLDER_PATH, filename)
        print(f"\n--- Reading file: {filename} ---")
        score = 0
        total = 0
        # Open and read the CSV file
        with open(file_path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            # Print each row
            for row in tqdm(reader):
                question, option_a, option_b, option_c, option_d, result = row
                user_prompt = f"Question: {question}, Option A: {option_a}, Option B: {option_b}, Option C: {option_c}, Option D: {option_d}"
               
                response = client.chat.completions.create(
                     model=MODEL_NAME,
                        messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                        ],
                        seed=256) # keep other parameters fixed                    )

                llm_output = response.choices[0].message.content
                total = total + 1

                if llm_output == result:
                    score = score + 1
                else:
                    print(f"Result not matched to output. Result: {result}, LLM_output: {llm_output}")
        results.append({'file_name': filename, "score": score, "percentage": round(score/total, 4)})

    # Save results as JSON
    with open("evaluation/results/llm_evaluation/base/o3.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)
        

print(results)
