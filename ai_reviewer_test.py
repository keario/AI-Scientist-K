# use this file to run ai reviewer!

import openai # ensure openai package is updated to latest version to access new models and features
from ai_scientist.perform_review import load_paper, perform_review
import json

client = openai.OpenAI()
model =  "gpt-5.4-2026-03-05" # see list AVAILABLE_LLMS in llm.py for available models

# Load paper from PDF file (raw text)
paper_txt = load_paper("/Users/macbookair/Downloads/2pg_paper.pdf") # put filepath
print("Finished loading paper. Length:", len(paper_txt))
#paper_txt=paper_txt[:200] #truncate for testing

# set num_reflections=1, num_fs_examples=0, num_reviews_ensemble=1, temperature=0.1, reasoning_effort="low" for faster/cheaper testing
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5, # number of reflections on generated review (recommended 5)
    num_fs_examples=1, # number of fewshot examples to provide in the prompt (set to 0 to disable fewshot prompting)
    num_reviews_ensemble=5, # number of opinions generated (recommended 5)
    temperature=0.1, # variability. not supported by the gpt-5 models
    reasoning_effort="low", # e.g. "none", "low", "medium", "high", "xhigh", only applicable for gpt-5 models
)

# Inspect review results
print("Full Review Results in JSON Format:")
print(json.dumps(review, indent=2))

# print(review["Overall"])    # Overall score (1-10)
# print(review["Decision"])   # 'Accept' or 'Reject'
# print(review["Weaknesses"]) # List of weaknesses (strings)
