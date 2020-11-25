import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from typing import List
import os
# from dumper import dump

model = joblib.load(f'{os.getcwd()}/model.sav')
app = FastAPI()


@app.get('/')
async def index():
    return {"greeting": "hello Babou"}


class Account(BaseModel):
    balance: int
    total_payed: int
    status: int


@app.post('/predictions')
def predictions(accounts: List[Account]):
    data = pd.DataFrame([account.__dict__ for account in accounts])
    # dump(data)
    return {'prediction': list(model.predict(data))}


if __name__ == '__main__':
    uvicorn.run(app,host = '0.0.0.0', port = 8080)