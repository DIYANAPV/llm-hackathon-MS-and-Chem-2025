# evaluate_with_transformers.py
import json
import csv
from pathlib import Path
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed # Removed concurrency imports
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Configuration ===
DATASET_FOLDER_PATH = Path("evaluation/dataset/evaluation_dataset")
PDF_FOLDER_PATH = Path("evaluation/original_papers")
RESULTS_PATH = Path("evaluation/results/mcq_evaluation_llm/original_papers/qwen3_4B_thinking_2507_fp8.json")
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507-FP8"  # change to the HF model or local path
MAX_WORKERS = 4  # This configuration variable is now unused but kept for reference

SYSTEM_PROMPT = """You will be given:
A reference PDF text of a scientific paper.
A question with four options (A, B, C, D). Answer the right option in a json format with "answer" as key and relevant option as value. Respond with exactly one of the following strings. "Option A", "Option B", "Option C", or "Option D". No punctuation, explanation, or extra text.

# Example:
Question: What is the capital of France?, OptionA. Berlin Option B. Madrid Option C. Paris Option D. Rome.
Output: {"answer": Option A}

Question: What is 2 + 2?, Option A. 3 Option B. 4 Option C. 5 Option D. 6
Output: {"answer": Option B}
"""

def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def find_matching_pdf(csv_path: Path) -> Path | None:
    candidate = PDF_FOLDER_PATH / f"{csv_path.stem}.pdf"
    return candidate if candidate.exists() else None

# === Load tokenizer + model (shard across GPUs) ===
print("Loading tokenizer and model. This can take a while...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder=None,  # optional: set to use CPU offloading
)
model.eval()
print("Model loaded.")

def predict_from_prompt(prompt: str, max_tokens: int = 32) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    # Use chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,        
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated part
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

def evaluate_file(file_path: Path, pdf_text: str):
    score = 0
    total = 0

    with file_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 6:
                continue

            question, option_a, option_b, option_c, option_d, correct_answer = row
            user_prompt = f"""PDF: {pdf_text}, Question:{question}, Options: A. {option_a} B. {option_b} C. {option_c} D. {option_d}. Respond with exactly one of: Option A, Option B, Option C, Option D."""

            try:
                llm_output = predict_from_prompt(user_prompt, max_tokens=6400).replace('"','').strip()
                total += 1
                if correct_answer in llm_output:
                    score += 1
                else:
                    print(f"[{file_path.name}] Line: {total} ❌ Expected: {correct_answer}, Got: {llm_output}")
            except Exception as e:
                print(f"[{file_path.name}] ⚠️ Error: {e}")

    accuracy = round(score / total, 4) if total else 0.0
    return {"file_name": file_path.name, "score": score, "percentage": accuracy}

def process_csv_pdf_pair(csv_path: Path):
    pdf_path = find_matching_pdf(csv_path)
    if not pdf_path:
        print(f"⚠️ No matching PDF found for {csv_path.name}")
        return {"file_name": csv_path.name, "score": 0, "percentage": 0.0}

    pdf_text = extract_pdf_text(pdf_path)
    print(f"📘 Loaded {pdf_path.name} ({len(pdf_text.split())} words)")
    return evaluate_file(csv_path, pdf_text)

def main():
    csv_files = sorted(DATASET_FOLDER_PATH.glob("*.csv"))
    results = []

    print(f"🧠 Evaluating {len(csv_files)} CSV–PDF pairs sequentially...\n")

    # Iterate sequentially instead of using a ThreadPoolExecutor
    for csv_path in tqdm(csv_files, desc="Evaluating files"):
        try:
            result = process_csv_pdf_pair(csv_path)
            results.append(result)
            # Write results periodically
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