import sys
import os
from typing import List
import numpy as np
from typing import Dict, Any
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if os.environ.get("USE_SMALL_DATASET") == None:
    os.environ['USE_SMALL_DATASET'] = "True"

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, '../../'))

from src.models.modelmanager import ModelManager

username = os.environ.get("USER") or os.environ.get("USERNAME")
print(username)

# Set the tracking URI
mlflow_tracking_uri = f"./experiments/mlruns"
if not os.path.exists(mlflow_tracking_uri):
    os.makedirs(mlflow_tracking_uri)

if not os.path.exists('./model'):
    os.makedirs('./model')

os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

# Initialize FastAPI
app = FastAPI()
manager = ModelManager()

# Define request model
class PredictionRequest(BaseModel):
    ensemble_predict: bool
    data: List[List[int]]

class ModelParams(BaseModel):
    param: Dict[str, Any]
    n_jobs: int
    cv: int

class HyperparameterGrid(BaseModel):
    use_small_dataset: bool
    grid_search_params: Dict[str, ModelParams]

@app.get("/")
async def get_root():
    try:
        # Train the models and log them to MLflow
        manager.train(os.environ.get("USE_SMALL_DATASET") == "True")
        return {"message": "Training completed and logged to MLflow"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/best_model_parameters")
async def get_best_model_parameters():
    try:        
        if manager.best_model is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        best_params = manager.best_model_params
        if best_params is None:
            raise HTTPException(status_code=400, detail="Best model parameters not found")
        
        response_model = dict()
        response_model["best_model"] = manager.best_model.get_best_params()
        for model in manager.models:
            response_model[model._name] = model.get_best_params()
        return response_model
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training")
async def training(request: HyperparameterGrid):
    try:
        print(f"Training request: {request}")
        # Trigger training process
        use_small_dataset = os.environ.get("USE_SMALL_DATASET") == "True"
        if request.use_small_dataset is not None:
            use_small_dataset = request.use_small_dataset
        
        manager.train(use_small_dataset, request.grid_search_params)

        return {"message": "Training completed, best model logged in MLflow."}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        ensemble_predict = False
        if request.ensemble_predict is not None:
            ensemble_predict = request.ensemble_predict
        
        X = np.array(request.data)
        prediction = manager.predict(X, ensemble_predict)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# def start_mlflow():
#     with open('mlflow_log.txt', 'w') as f:
#         subprocess.Popen(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"], stdout=f, stderr=f)

if __name__ == "__main__":
    print('Calling the __main__ method....')
    # Enable autologging    
    mlflow.autolog()
    mlflow.keras.autolog()
    mlflow.sklearn.autolog()

    # start_mlflow()
    # mlflow.set_tracking_uri(f"file:{mlrun_tracking_uri}")

    mlflow.set_experiment(f"mlops-dynamos-experiment-{username}")
    with mlflow.start_run(nested=True):
        print('Started mlflow run command')
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8001, reload=True)
