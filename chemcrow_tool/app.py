from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="ChemCrow API")


try:
    from chemcrow.agents import ChemCrow
    CHEMCROW_AVAILABLE = True
    print("ChemCrow loaded successfully")
except ImportError as e:
    CHEMCROW_AVAILABLE = False
    print(f"ChemCrow import failed: {e}")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "ChemCrow API", "chemcrow_available": CHEMCROW_AVAILABLE}

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "chemcrow_available": CHEMCROW_AVAILABLE,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/query")
def chemistry_query(request: QueryRequest):
    if not CHEMCROW_AVAILABLE:
        raise HTTPException(status_code=500, detail="ChemCrow not available")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    
    try:
        chem_model = ChemCrow(
            model="gpt-4o-mini",           # Main model
            tools_model="gpt-4o-mini",     # Tools model (this was causing the error) changed from old gpt3.5 to this 
            temp=0.1,
            streaming=False,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        result = chem_model.run(request.query)
        
        return {
            "success": True,
            "query": request.query,
            "result": str(result)
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": request.query,
            "error": str(e)
        }
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)