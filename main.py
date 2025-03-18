from fastapi import FastAPI,UploadFile,File
from pathlib import Path
import shutil
import torch
import timm
from torchvision import transforms
import torch
from skin import predict_image


app = FastAPI(
    version = "1.0.1",
    title ="Skin Condition"
)

@app.get("/")
async def root():
    return{"message":"Welcome, we are alive"}

@app.post("/predict/")
async def predict(file:UploadFile = File(...)):
    
    temp_file = Path(f"temp_{file.filename}")
    
    with temp_file.open(mode="wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
        
# Prediction 
    prob,preds = predict_image(temp_file)
    
#delete the file after use 
    
    temp_file.unlink()
    
    return{f"The predicted skin condition is: {preds} with probability of: {round(prob,3)}"}