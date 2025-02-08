import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.join(os.path.dirname(__file__), '../../experiments'))

import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException

app = FastAPI()



@app.get("/")
async def root():
    try:            
        return { "mlflow-running-id": f"{mlflow_run_id}" }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Enable autologging    
    mlflow.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Set the experiment name 
    mlflow.set_experiment("mlops-dynamos-experiment")
    # Start a new MLflow run
    with mlflow.start_run():
        mlflow_run_id = mlflow.active_run().info.run_id
        # Run the FastAPI application
        uvicorn.run(app, host="0.0.0.0", port=8001)