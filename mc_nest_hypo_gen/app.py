from fastapi import FastAPI
from pydantic import BaseModel
from config.enums import SelectionPolicy, InitialStrategy
from src.mc_nest import MC_NEST_gpt4o
import uvicorn
app = FastAPI(title="MC-NEST Hypothesis API")

# Request schema
class HypothesisRequest(BaseModel):
    user_prompt: str
    background_information: str
    # rollouts: int = 2
    # selection_policy: SelectionPolicy = SelectionPolicy.GREEDY
    # initialize_strategy: InitialStrategy = InitialStrategy.ZERO_SHOT

# Response schema
class HypothesisResponse(BaseModel):
    best_hypothesis: str

@app.post("/generate", response_model=HypothesisResponse)
def generate_hypothesis(request: HypothesisRequest):
    print("Received request:", request)
    mc_nest = MC_NEST_gpt4o(
        user_prompt=request.user_prompt,
        background_information=request.background_information,
        max_rollouts=2,
        selection_policy=2,
        initialize_strategy=1
    )

    best_hypothesis = mc_nest.run()
    return HypothesisResponse(best_hypothesis=best_hypothesis)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0", port=8000)