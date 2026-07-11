from fastapi import FastAPI
from pydantic import BaseModel
import torch

from inference import load_model, translate

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tok_en, tok_fr, config = load_model(
    "checkpoints",
    "tokenizers",
    device,
)

class Request(BaseModel):
    text: str


@app.post("/translate")
def translate_api(req: Request):

    prediction = translate(
        req.text,
        model,
        tok_en,
        tok_fr,
        config,
        device,
    )

    return {
        "english": req.text,
        "french": prediction,
    }