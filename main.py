# main.py
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
from fastapi.responses import RedirectResponse

app = FastAPI(title="House Price Predictor")

MODEL_PATH = "house_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found in app root.")

model = joblib.load(MODEL_PATH)

class Input(BaseModel):
    data: Optional[List[float]] = [8.3252, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input):
    arr = np.array(input.data, dtype=float).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": float(pred[0])}

# --- Gradio UI ---
def gr_predict(medinc, houseage, averooms, avebedrms, population, aveoccup, lat, long):
    arr = np.array([medinc, houseage, averooms, avebedrms, population, aveoccup, lat, long]).reshape(1, -1)
    return float(model.predict(arr)[0])

with gr.Blocks() as demo:
    gr.Markdown("# House Price Predictor")
    with gr.Row():
        f1 = gr.Number(value=8.3252, label="MedInc")
        f2 = gr.Number(value=41.0, label="HouseAge")
        f3 = gr.Number(value=6.98, label="AveRooms")
        f4 = gr.Number(value=1.02, label="AveBedrms")
    with gr.Row():
        f5 = gr.Number(value=322.0, label="Population")
        f6 = gr.Number(value=2.55, label="AveOccup")
        f7 = gr.Number(value=37.88, label="Latitude")
        f8 = gr.Number(value=-122.23, label="Longitude")
    btn = gr.Button("Predict")
    out = gr.Textbox(label="Predicted value")
    btn.click(fn=gr_predict, inputs=[f1,f2,f3,f4,f5,f6,f7,f8], outputs=out)

# Mount Gradio on the FastAPI app at root path
app = gr.mount_gradio_app(app, demo, path="/")

# Optional: redirect to root
@app.get("/")
def root():
    return RedirectResponse(url="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
