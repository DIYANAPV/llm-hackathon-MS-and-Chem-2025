import json
from tqdm import tqdm

from openai import OpenAI

client = OpenAI()



if __name__ == "__main__":

    models = ["o4-mini"]
    paper_titles = [
        "Topographically selective atomic layer deposition within trenches enabled by an amorphous carbon inhibition layer",
        "Subnano Al2O3 Coatings for Kinetics and Stability Optimization of LiNi0.83Co0.12Mn0.05O2 via O3-Based Atomic Layer Deposition",
        "Molecular Design in Area-Selective Atomic Layer Deposition: Understanding Inhibitors and Precursors",
        "Catalysts Design and Atomistic Reaction Modulation byAtomic Layer Deposition for Energy Conversion and StorageApplications",
        "Spatial Atomic Layer Deposition for Energy and Electronic Devices"
    ]

    for model in tqdm(models):
        results = []

        for paper in paper_titles:
            response = client.responses.create(
                model=model,
                instructions=(
                    "Provide me a meta data of this paper regarding authors, "
                    "year that the paper is published, DOI of the paper and a small "
                    "summary of the paper. DO NOT USE ANY TOOLS or WEB SEARCH. Provide it in the following JSON format: " \
                    "{ 'authors': [list of authors], 'year': year, 'DOI': 'DOI of the paper', 'summary': 'small summary of the paper' }"
                ),
                input=paper,
            )

            # Collect structured data
            results.append({
                "paper_title": paper,
                "response": response.output_text
            })

    # Save results into a JSON file named after the model
        filename = f"{model}_papers_metadata.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved metadata for model {model} into {filename}")
