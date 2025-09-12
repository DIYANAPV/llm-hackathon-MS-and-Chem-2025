# source = 'https://github.com/SamuelSchmidgall/AgentLaboratory/blob/main/agents.py'
import re
import json
from openai import OpenAI
import pymupdf4llm
from tqdm import tqdm

client = OpenAI()


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found



def get_score(latex, reward_model_llm, reviewer_type=None, attempts=3):
    e = str()
    for _ in range(attempts):
        try:
            # todo: have a reward function here
            # template inherited from the AI Scientist (good work on this prompt Sakana AI team :D)
            template_instructions = """
            Respond in the following format:

            THOUGHT:
            <THOUGHT>

            REVIEW JSON:
            ```json
            <JSON>
            ```

            In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
            Detail your high-level arguments, necessary choices and desired outcomes of the review.
            Do not make generic comments here, but be specific to your current paper.
            Treat this as the note-taking phase of your review.

            In <JSON>, provide the review in JSON format with the following fields in the order:
            - "Summary": A summary of the paper content and its contributions.
            - "Strengths": A list of strengths of the paper.
            - "Weaknesses": A list of weaknesses of the paper.
            - "Originality": A rating from 1 to 4 (low, medium, high, very high).
            - "Quality": A rating from 1 to 4 (low, medium, high, very high).
            - "Clarity": A rating from 1 to 4 (low, medium, high, very high).
            - "Significance": A rating from 1 to 4 (low, medium, high, very high).
            - "Questions": A set of clarifying questions to be answered by the paper authors.
            - "Limitations": A set of limitations and potential negative societal impacts of the work.
            - "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
            - "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Overall": A rating from 1 to 10 (very strong reject to award quality).
            - "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
            - "Decision": A decision that has to be one of the following: Accept, Reject.

            For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
            This JSON will be automatically parsed, so ensure the format is precise.
            """
            neurips_form = ("""
                ## Review Form
                Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
                When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public. 

                1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary. This is also not the place to paste the abstract—please provide the summary in your own understanding after reading.
                2. Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper. A good mental framing for strengths and weaknesses is to think of reasons you might accept or reject the paper. Please touch on the following dimensions:
                    Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
                    Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
                    Significance: Are the results impactful for the community? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance our understanding/knowledge on the topic in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
                    Originality: Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? Is it clear how this work differs from previous contributions, with relevant citations provided? Does the work introduce novel tasks or methods that advance the field? Does this work offer a novel combination of existing techniques, and is the reasoning behind this combination well-articulated? As the questions above indicates, originality does not necessarily require introducing an entirely new method. Rather, a work that provides novel insights by evaluating existing methods, or demonstrates improved efficiency, fairness, etc. is also equally valuable.

                3. Quality: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the quality of the work.
                    4 excellent
                    3 good
                    2 fair
                    1 poor
                4. Clarity: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the clarity of the paper.
                    4 excellent
                    3 good
                    2 fair
                    1 poor
                5. Significance: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the significance of the paper.
                    4 excellent
                    3 good
                    2 fair
                    1 poor
                6. Originality: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the originality of the paper.
                    4 excellent
                    3 good
                    2 fair
                    1 poor
                7. Questions: Please list up and carefully describe questions and suggestions for the authors, which should focus on key points (ideally around 3–5) that are actionable with clear guidance. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. You are strongly encouraged to state the clear criteria under which your evaluation score could increase or decrease. This can be very important for a productive rebuttal and discussion phase with the authors.
                8. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If so, simply leave “yes”; if not, please include constructive suggestions for improvement. In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.
                9. Overall: Please provide an "overall score" for this submission. Choices:
                    6: Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
                    5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
                    4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
                    3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
                    2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
                    1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations
                10. Confidence:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation.  Choices
                    5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
                    4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
                    3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                    2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                    1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
                11. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review.
                            
                  You must make sure that all sections are properly created: abstract, introduction, methods, results, and discussion. Points must be reduced from your scores if any of these are missing.
                """ + template_instructions)
            if reviewer_type is None: reviewer_type = ""
            system_prompt = (
                      "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
                      f"Be critical and cautious in your decision. {reviewer_type}\n"
                  ) + neurips_form
            user_prompt= f"The following text is the research latex that the model produced: \n{latex}\n\n"
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            response = client.chat.completions.create(
                model=reward_model_llm,
                messages=messages,
                seed=256)

            scoring = response.choices[0].message.content

        
            review_json = extract_json_between_markers(scoring)

            review_json["Overall"] = float(review_json["Overall"])
            review_json["Soundness"] = float(review_json["Soundness"])
            review_json["Confidence"] = float(review_json["Confidence"])
            review_json["Contribution"] = float(review_json["Contribution"])
            review_json["Presentation"] = float(review_json["Presentation"])
            review_json["Clarity"] = float(review_json["Clarity"])
            review_json["Originality"] = float(review_json["Originality"])
            review_json["Quality"] = float(review_json["Quality"])
            review_json["Significance"] = float(review_json["Significance"])

            return review_json, scoring, True
        except Exception as e:
            print(e)
            return None, str(e), False
    return None, e, False

class ReviewersAgent:
    def __init__(self, model="gpt-5-mini"):
        self.model = model

    def inference(self, report):
        reviewer_1 = "You are a harsh but fair reviewer for a position paper who expects the proposed idea to be feasible and grounded in realistic methodology, while still providing experimental or conceptual evidence that the approach could work in practice." # novel idea
        review_1 = get_score( latex=report, reward_model_llm=self.model, reviewer_type=reviewer_1)

        reviewer_2 = "You are a harsh and critical but fair reviewer for a position paper who evaluates whether the idea is communicated with clarity, well-framed within the existing literature, and presented in a coherent and professional format." # impactful idea
        review_2 = get_score(latex=report, reward_model_llm=self.model, reviewer_type=reviewer_2)

        reviewer_3 = "You are a harsh but fair open-minded reviewer for a position paper who values genuine novelty and originality, and looks for creative directions that have not been explored before in the literature." # 
        review_3 = get_score(latex=report, reward_model_llm=self.model, reviewer_type=reviewer_3)

        return f"Reviewer #1:\n{review_1}, \nReviewer #2:\n{review_2}, \nReviewer #3:\n{review_3}"


if __name__ == "__main__":
    # Example usage
    
    pdf_path = "/home/gsasikiran/tib/code/utils/llm_hackathon/generated_papers/o3/spatial_atomic_layer_deposition.pdf"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    review_json, scoring, success = get_score(latex=md_text, reward_model_llm="gpt-5", reviewer_type="You are a harsh but fair open-minded reviewer for a position paper who values genuine novelty and originality, and looks for creative directions that have not been explored before in the literature.")
    if success:
        json_path = "/home/gsasikiran/tib/code/utils/llm_hackathon/generated_papers/o3/spatial_atomic_layer_deposition" + ".json"
        with open(json_path, "w") as f:
            json.dump(review_json, f, indent=4)
        print(f"Successfully processed {pdf_path} for reviewer_3 and saved results to {json_path}")
    else:
        print(f"Failed to process {pdf_path} for reviewer_3")
        print(f"Error: {scoring}")
        print(f"Error: {scoring}")

