# Handwritten Digit Recognition

This project implements an Ensemble Model to classify handwritten digits from the MNIST dataset. The project includes a FastAPI application for serving the model, MLflow for tracking experiments, and Docker for containerization.

This `README.md` provides a comprehensive guide on setting up the server, API details, Docker setup, and how to run the code manually.

## Prerequisites

- Docker Desktop installed on your machine
- Python 3.9 installed on your machine
- pip (Python package installer) installed
- conda installed on your machine

## Project Structure

Here is the README.md content for your project:

  ├── dataset/ 
  ├── experiments/ 
  ├── model/ 
  ├── notebook/ 
  ├── src/ 
  ├── tests/ 
  ├── .dockerignore /
  ├── .gitignore /
  ├── Dockerfile /
  ├── README.md /
  ├── requirements.txt /
  └── supervisord.conf

## How to Run this Project

### 1. Using Docker image:
```sh
docker pull kpunit03/mlops-dynamos-mnist
```

### 2. Using Docker

#### Build the Docker Image

```sh
docker build -t mlops-dynamos-mnist .
```

#### Run the Docker Container

```sh
docker run --name mlops-dynamos-mnist -p 8001:8001 -p 5000:5000 mlops-dynamos-mnist
```

#### Open a Shell in the Docker Container
```sh
docker exec -it mlops-dynamos-mnist sh
# OR
docker exec -it mlops-dynamos-mnist /bin/bash
```

### 3. Running Manually

#### Clone the Repository

```sh
git clone https://github.com/kmrpunit/mlops_dynamos_mnist.git
cd mlops_dynamos_mnist
```

#### Create a Virtual Environment

```sh
conda create --name mlops-dynamos-env python=3.9
conda activate mlops-dynamos-env
```

#### Install Dependencies

```sh
pip install -r requirements.txt
```

#### Run the Application

```sh
uvicorn src.main:app --host 0.0.0.0 --port 8001
```

#### Access MLflow

```sh
mlflow ui
```

Open your browser and go to `http://127.0.0.1:5000` to access the MLflow UI.

## API Endpoints

### Get Root
#### Endpoint: GET /
- **URL:** `http://127.0.0.1:8001`
- **Method:** `GET`
- **Response:** Training success or Error message in JSON format

**Description**: Starts training with default parameters

### Get Best Model Parameters
#### Endpoint: GET /best_model_parameters
- **URL:** `http://127.0.0.1:8001/best_model_parameters`
- **Method:** `GET`
- **Response:** Training success or Error message in JSON format

**Description**: Returns the parameters of the best model alongwith the best parameters of all other model.

**Example Request**:
Refer [test_best_model_parameters.py](tests/test_best_model_parameters.py)

**Example Response**
```json
{
  "best_model": {
    "name": "cnn",
    "best_params": {
      "batch_size": 32,
      "epochs": 10,
      "optimizer": "adam"
    },
    "accuracy": 0.9798868511726132,
    "test_accuracy": 0.979,
    "test_loss": 0.07556225669369931,
    "roc_auc_score": 0.9995352310030123,
    "recall": 0.9784272991388274
  },
  "fcnn": {
    "name": "fcnn",
    "best_params": {
      "batch_size": 32,
      "epochs": 10,
      "optimizer": "adam"
    },
    "accuracy": 0.9527726649623949,
    "test_accuracy": 0.955,
    "test_loss": 0.16779326924600618,
    "roc_auc_score": 0.9986605898496113,
    "recall": 0.9545408450468887
  },
  "xgboost": {
    "name": "xgboost",
    "best_params": {
      "learning_rate": 0.3,
      "max_depth": 3,
      "n_estimators": 100
    },
    "accuracy": 0.9462169982586789,
    "test_accuracy": 0.945,
    "test_loss": 0.16080760274660577,
    "roc_auc_score": 0.9982005293930204,
    "recall": 0.9441362301471965
  },
  "rf": {
    "name": "rf",
    "best_params": {
      "max_depth": null,
      "min_samples_split": 5,
      "n_estimators": 100
    },
    "accuracy": 0.9473278500240822,
    "test_accuracy": 0.951,
    "test_loss": 0.3630722886739527,
    "roc_auc_score": 0.9982682820589386,
    "recall": 0.9502418276888569
  },
  "svm": {
    "name": "svm",
    "best_params": {
      "C": 10,
      "gamma": 0.01,
      "kernel": "rbf"
    },
    "accuracy": 0.9654408506539217,
    "test_accuracy": 0.971,
    "test_loss": 0.09458629708145037,
    "roc_auc_score": 0.9995358181438228,
    "recall": 0.9703796089260954
  },
  "cnn": {
    "name": "cnn",
    "best_params": {
      "batch_size": 32,
      "epochs": 10,
      "optimizer": "adam"
    },
    "accuracy": 0.9798868511726132,
    "test_accuracy": 0.979,
    "test_loss": 0.07556225669369931,
    "roc_auc_score": 0.9995352310030123,
    "recall": 0.9784272991388274
  }
}
```

### Train Model
#### Endpoint: POST /training
- **URL:** `http://127.0.0.1:8001/training`
- **Method:** `POST`
- **Request Body:** JSON containing the grid search parameters of each model along with cv and n_jobs value
- **Response:** JSON containing the training success or Error message

**Description**: Trains the model with the provided data.

**Example Request**:
Refer [test_training.py](tests/test_training.py)

```json
{
    "use_small_dataset": true,
    "grid_search_params": {
        "fcnn": {
            "param": {
                "batch_size": [
                    32,
                    64,
                    128
                ],
                "epochs": [
                    5,
                    10
                ],
                "optimizer": [
                    "adam",
                    "rmsprop"
                ]
            },
            "n_jobs": -1,
            "cv": 3
        },
        "svm": {
            "param": {
                "C": [
                    1,
                    10,
                    100
                ],
                "gamma": [
                    0.1,
                    0.01
                ],
                "kernel": [
                    "rbf"
                ]
            },
            "n_jobs": -1,
            "cv": 3
        },
        "rf": {
            "param": {
                "n_estimators": [
                    50,
                    75,
                    100
                ],
                "max_depth": [
                    null,
                    10,
                    20
                ],
                "min_samples_split": [
                    2,
                    5,
                    10
                ]
            },
            "n_jobs": -1,
            "cv": 3
        },
        "xgboost": {
            "param": {
                "max_depth": [
                    3,
                    6
                ],
                "learning_rate": [
                    0.1,
                    0.3
                ],
                "n_estimators": [
                    50,
                    100
                ]
            },
            "n_jobs": -1,
            "cv": 3
        },
        "cnn": {
            "param": {
                "batch_size": [
                    32,
                    64,
                    128
                ],
                "epochs": [
                    5,
                    10
                ],
                "optimizer": [
                    "adam",
                    "rmsprop"
                ]
            },
            "n_jobs": -1,
            "cv": 3
        }
    }
}
```

** Example Response: **
```json
{
  "message": "Training completed and logged to MLflow"
}
```

### Predict Digit

- **URL:** `http://127.0.0.1:8001/predict`
- **Method:** `POST`
- **Request Body:** JSON containing the image data 
- **Response:** JSON containing the predicted digit

**Example Request**:
Refer: 
1. [test_ensemble_predict.py](tests/test_ensemble_predict.py)
2. [test_predict](tests/test_predict.py)

**Example Response**

```json
{"prediction": [7, 2]}
```

## Working of the Project
1. **Data Preprocessing**: The MNIST dataset is preprocessed to be used by different models.
2. **Model Training**: Multiple models(CNN(KerasClassifier), FCCN(KerasClassifier), SVM, RandomForest and XGBoost) are trained and their parameters are tuned using hyperparameter tuning techniques.
3. **Ensemble Model**: An ensemble model is created using the trained models.
4. **Model Evaluation**: The ensemble model is evaluated using various metrics.
5. **API**: A FastAPI application is used to serve the model and provide endpoints for training and prediction.
6. **MLflow**: MLflow is used to track experiments and log metrics.
7. **Docker**: The entire application is containerized using Docker for easy deployment.

## Final Deliverable
* A GitHub repository with:
  * The ML model code.
  * Hyperparameter tuning implementation.
  * MLflow tracking integration.
  * A FastAPI application serving the model.
  * A Dockerfile for deployment.
  * A working Docker container running the complete ML pipeline.
