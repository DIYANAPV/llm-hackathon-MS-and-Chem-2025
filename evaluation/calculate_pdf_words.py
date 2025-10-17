# evaluate_with_transformers.py
from pathlib import Path
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed # Removed concurrency imports
from PyPDF2 import PdfReader


# === Utility Functions ===
def calculate_pdf_words(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    text = "\n".join(text)
    print(f"📘 Loaded {pdf_path.name} ({len(text.split())} words)")
    return len(text.split())


if __name__ == "__main__":
    PDF_FOLDER_PATH = Path("evaluation/generated_papers/gpt-5")  # folder with PDFs
    pdf_files = sorted(PDF_FOLDER_PATH.glob("*.pdf"))
    pdf_word_counts = {}

    print(f"🧠 Calculating word counts for {len(pdf_files)} PDFs...\n")

    for pdf_path in tqdm(pdf_files, desc="Calculating PDF word counts"):
        try:
            word_count = calculate_pdf_words(pdf_path)
            pdf_word_counts[pdf_path.name] = word_count
        except Exception as e:
            print(f"[{pdf_path.name}] ❌ Failed: {e}")

    # Print summary
    print("\n✅ PDF Word Count Summary:")
    for file_name, count in pdf_word_counts.items():
        print(f"{file_name}: {count} words")