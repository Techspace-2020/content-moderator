from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel

app = FastAPI()
handler = Mangum(app)

# Define the expected input structure
class FakeNewsRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Fake News Detection API is running!"}

@app.post("/predict")
async def predict(request: FakeNewsRequest):
    # Dummy logic for now (since phi.agent was missing)
    if "fake" in request.text.lower():
        result = "Likely Fake News"
    else:
        result = "Likely Real News"
    
    return {
        "input": request.text,
        "prediction": result
    }