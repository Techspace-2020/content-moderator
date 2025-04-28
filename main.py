from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def read_root():
    return {"message":"This is first fast api project"}

@app.get("/test")
async def sample():
    return {"message":"Testing purpose"}