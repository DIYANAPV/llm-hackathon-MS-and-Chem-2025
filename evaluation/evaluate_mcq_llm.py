import json
import csv
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DATASET_FOLDER_PATH = Path("evaluation/dataset/evaluation_dataset")
RESULTS_PATH = Path("evaluation/results/mcq_evaluation_llm/base/test.json")
MODEL_NAME = "o4-mini"  # or "gpt-5"
MAX_WORKERS = 4  # Increase if your rate limits allow (4–6 is usually safe)

SYSTEM_PROMPT = """You will be provided with a question and four options. 
You have to provide the correct answer as a single word string: 
'Option A', 'Option B', 'Option C', or 'Option D'."""

# Initialize client (thread-safe)
client = OpenAI()


def evaluate_file(file_path: Path):
    """Evaluate a single CSV file and return its score summary."""
    score = 0
    total = 0

    with file_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip header

        for i in range(5):  # Limit to first 5 questions for quick testing
            row = next(reader, None)
            if len(row) < 6:
                continue
            question, option_a, option_b, option_c, option_d, correct_answer = row

            user_prompt = (
                f"Question: {question}\n"
                f"Option A: {option_a}\n"
                f"Option B: {option_b}\n"
                f"Option C: {option_c}\n"
                f"Option D: {option_d}"
            )

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    seed=256
                )
                llm_output = response.choices[0].message.content.strip().replace('"', '')

                total += 1
                if llm_output == correct_answer:
                    score += 1
                else:
                    print(f"[{file_path.name}] Mismatch → Expected: {correct_answer}, Got: {llm_output}")

            except Exception as e:
                print(f"[{file_path.name}] Error: {e}")

    accuracy = round(score / total, 4) if total else 0.0
    return {"file_name": file_path.name, "score": score, "percentage": accuracy}


def main():
    results = []
    csv_files = sorted(DATASET_FOLDER_PATH.glob("*.csv"))

    print(f"🧠 Evaluating {len(csv_files)} CSV files using {MAX_WORKERS} threads...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_file, file): file for file in csv_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating files"):
            file_path = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Save partial results (for safety)
                with RESULTS_PATH.open("w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)

            except Exception as e:
                print(f"[{file_path.name}] Failed with error: {e}")

    print("\n✅ Evaluation Complete. Summary:")
    for r in results:
        print(f"{r['file_name']}: {r['percentage'] * 100:.2f}% accuracy")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
