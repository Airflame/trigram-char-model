import uvicorn
from trigram import trigram
from fastapi import FastAPI

app = FastAPI()

@app.get("/generate")
def generate(word_length: int = None, word_count: int = 100, gender: str = None):
    return trigram(word_length=word_length, word_count=word_count, gender=gender)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)