import json
import csv
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader

# === Configuration ===
DATASET_FOLDER_PATH = Path("evaluation/dataset/evaluation_dataset")
PDF_FOLDER_PATH = Path("evaluation/original_papers")  # folder with corresponding PDFs
RESULTS_PATH = Path("evaluation/results/mcq_evaluation_llm/original_papers/gpt-5.json")
MODEL_NAME = "gpt-5"  # or "o4-mini", "gpt-5"
MAX_WORKERS = 8  # Tune based on rate limits and system resources

SYSTEM_PROMPT = """You will be provided with:
1. A reference document
2. A question and four options

You must choose the correct answer as one of:
'Option A', 'Option B', 'Option C', or 'Option D'.

Only provide one of 'Option A', 'Option B', 'Option C', or 'Option D' as a single word string."""

client = OpenAI()

# === Utility Functions ===
def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def find_matching_pdf(csv_path: Path) -> Path | None:
    """Find a PDF file with the same stem as the CSV file."""
    candidate = PDF_FOLDER_PATH / f"{csv_path.stem}.pdf"
    return candidate if candidate.exists() else None


def evaluate_file(file_path: Path, pdf_text: str):
    """Evaluate all questions in a CSV file using its corresponding PDF text."""
    score = 0
    total = 0

    with file_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 6:
                continue

            question, option_a, option_b, option_c, option_d, correct_answer = row

            user_prompt = f"""
Reference Document:
{pdf_text}  # truncated to avoid context overflow

Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
"""

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
                    print(f"[{file_path.name}] Line number: {total} ❌ Expected: {correct_answer}, Got: {llm_output}")

            except Exception as e:
                print(f"[{file_path.name}] ⚠️ Error: {e}")

    accuracy = round(score / total, 4) if total else 0.0
    return {"file_name": file_path.name, "score": score, "percentage": accuracy}


def process_csv_pdf_pair(csv_path: Path):
    """Process a single CSV–PDF pair."""
    pdf_path = find_matching_pdf(csv_path)
    if not pdf_path:
        print(f"⚠️ No matching PDF found for {csv_path.name}")
        return {"file_name": csv_path.name, "score": 0, "percentage": 0.0}

    pdf_text = extract_pdf_text(pdf_path)
    print(f"📘 Loaded {pdf_path.name} ({len(pdf_text.split())} words)")

    return evaluate_file(csv_path, pdf_text)


# === Main Runner ===
def main():
    csv_files = sorted(DATASET_FOLDER_PATH.glob("*.csv"))
    results = []

    print(f"🧠 Evaluating {len(csv_files)} CSV–PDF pairs using {MAX_WORKERS} threads...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_csv_pdf_pair, file): file for file in csv_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating files"):
            csv_path = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Save partial progress
                with RESULTS_PATH.open("w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)

            except Exception as e:
                print(f"[{csv_path.name}] ❌ Failed: {e}")

    print("\n✅ Evaluation Complete. Summary:")
    for r in results:
        print(f"{r['file_name']}: {r['percentage'] * 100:.2f}% accuracy")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
