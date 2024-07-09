from typing import Union
from fastapi import FastAPI, Request
import os
# İgnore Warnings
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# Load environment variables from .env file
import pickle
cwd = os.getcwd()
file_name = cwd+"/model_svm.pkl"
file_name2 = cwd+"/model_ekedc.pkl"


app = FastAPI()


@app.get("/")
def read_root():
    
    return {"Hello": "World"}


model = pickle.load(open(file_name, "rb"))
scaler = model.named_steps['standardscaler']
def predict_next_hours(hours,scaler, model):
  former = [4.003, 4.203, 5.783, 6.836, 7.401, 4.871, 4.192, 5.152, 4.38, 4.867, 5.295, 4.101, 8.404, 8.148, 4.657]
  final = []
  for hour in range(hours):
    reshaped_input = np.array(former).reshape(1, -1)
    next_prediction = model.named_steps['svr'].predict(reshaped_input).flatten()[0] #model.predict(np.array(former))
    final.append(next_prediction)
    former.append(next_prediction)
    former = former[1:]
    #print(former)
  return model.named_steps['standardscaler'].inverse_transform(np.array(final).reshape(1, -1)).flatten()

@app.post("/predict_sa")
async def get_prediction(request: Request):
    # print(prediction)
    message = await request.json()
    
    #print(category["gender"].index(message["gender"]))
    hours =  message["hours"]
    prediction = predict_next_hours(hours,scaler, model)

    print(prediction)
    result = {"response":prediction.tolist()}
    return result #await request.json()



model_ekedc = pickle.load(open(file_name2, "rb"))
scaler_ekedc = model_ekedc.named_steps['standardscaler']
def predict_next_hours_ekedc(hours,scaler, model):
  former = [155.6,155.90000000000003,193.6,195.2,196.3,145.2,171.4,172.5,153.7,128.4,140.90000000000003,147.9,167.8,168.8,152.9]
  final = []
  for hour in range(hours):
    reshaped_input = np.array(former).reshape(1, -1)
    next_prediction = model.named_steps['svr'].predict(reshaped_input).flatten()[0] #model.predict(np.array(former))
    final.append(next_prediction)
    former.append(next_prediction)
    former = former[1:]
    #print(former)
  return model.named_steps['standardscaler'].inverse_transform(np.array(final).reshape(1, -1)).flatten()


@app.post("/predict_ekedc")
async def get_prediction(request: Request):
    # print(prediction)
    message = await request.json()
    
    #print(category["gender"].index(message["gender"]))
    hours =  message["hours"]
    prediction = predict_next_hours(hours,scaler_ekedc, model_ekedc)

    print(prediction)
    result = {"response":prediction.tolist()}
    return result #await request.json()