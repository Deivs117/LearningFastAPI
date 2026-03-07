from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

app = FastAPI()
prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    if item_id in prime_numbers:
        return {"item_id": item_id, "q": q, "message": "This is a prime number!"}
    return {"item_id": item_id, "q": q,  "message": "This is not a prime number."}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}