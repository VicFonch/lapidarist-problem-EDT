from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
import keras

from pipline import pipline


data = pd.read_csv('data/cleaned_data/cleaned_diamonds.csv')
data = data.drop(columns=['x', 'y', 'z', 'latitude', 'longitude', 'price'])
print(data.head())
preprocessor = pipline(data)
preprocessor.fit(data)

model = keras.models.load_model('models/mlp/diamonds_model_mlp.keras')

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    carat: float = Form(default=0.23),
    cut: str = Form(default="Ideal"),
    color: str = Form(default="E"),
    clarity: str = Form(default="SI2"),
    depth: float = Form(default=61.5),
    table: float = Form(default=55.0)
):
    input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table]],
                              columns=["carat", "cut", "color", "clarity", "depth", "table"])

    input_data_prep = preprocessor.transform(input_data)
    prediction = model.predict(input_data_prep)[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)