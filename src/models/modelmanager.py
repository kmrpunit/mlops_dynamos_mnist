import os
import pickle
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from src.models.ensemblemodel import EnsembleModel
from src.models.fcnn import NeuralNetworkModel, build_cnn
from src.models.rf import RandomForestModel
from src.models.svm import SVMModel
from src.models.xgboost import XGBoostModel
from src.preprocess.preprocess import ReadAndLoadMnistData
from scikeras.wrappers import KerasClassifier

# Dataset paths
dataset_dir = "./dataset"
train_file = f"{dataset_dir}/mnist_train.csv"
test_file = f"{dataset_dir}/mnist_test.csv"
input_shape = 28 * 28  # Flattened MNIST images


class ModelManager:

    default_grid_search_cv_params = {
        'fcnn': { 
                # Hyperparameters for FCNN
                'param' : { 
                'batch_size': [32, 64, 128],
                'epochs': [5, 10],
                'optimizer': ['adam', 'rmsprop']
                }, 
                'n_jobs' : -1, 
                'cv' : 3
            },
        'svm': { 
                # Hyperparameters for SVM
                'param' : {
                    'C': [1, 10, 100],
                    'gamma': [0.1, 0.01],
                    'kernel': ['rbf']
                }, 
                'n_jobs' : -1, 
                'cv' : 3
            },
        'rf': { 
                # Hyperparameters for Random Forest
                'param' : { 
                'n_estimators': [50, 75, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
                }, 
                'n_jobs' : -1,
                'cv' : 3
            },
        'xgboost': { 
                # Hyperparameters for XGBoost                
                'param' : {
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.3],
                    'n_estimators': [50, 100]
                }, 
                'n_jobs' : -1, 
                'cv' : 3
            },
        'cnn': {
            # Hyperparameters for CNN
            'param': {
                'batch_size': [32, 64, 128],
                'epochs': [5, 10],
                'optimizer': ['adam', 'rmsprop']
            },
            'n_jobs': -1,
            'cv': 3
        }
    }

    def __init__(self):
        print("ModelManager: __init__")
        self.trainers = []
        self.models = []
        self.best_model = None
        self.best_model_params = None
        self.best_model_test_accuracy = 0.0
        self.best_model_score = 0.0
        self.best_model_recall = 0.0
        self.best_model_cm_filename = None
        self.best_model_cv_results_filename = None
        self.best_model_filename = f"./model/best_model.pkl"
        self.ensemble_model_filename = f"./model/ensemble_model.pkl"
        self.ensemble_model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # Load and preprocess dataset
        with ReadAndLoadMnistData(train_file, test_file) as mnist_data:
            self.trainImagesLarge, self.trainLabelsLarge, self.trainImagesSmall, self.trainLabelsSmall = mnist_data.load_data()

            # Convert DataFrames to Numpy arrays if they are DataFrames
            if isinstance(self.trainImagesLarge, pd.DataFrame):
                print("ModelManager: Converting trainImagesLarge to numpy array")
                self.trainImagesLarge = self.trainImagesLarge.to_numpy()

            if isinstance(self.trainLabelsLarge, pd.DataFrame):
                print("ModelManager: Converting trainLabelsLarge to numpy array")
                self.trainLabelsLarge = self.trainLabelsLarge.to_numpy()

            if isinstance(self.trainImagesSmall, pd.DataFrame):
                print("ModelManager: Converting trainImagesSmall to numpy array")
                self.trainImagesSmall = self.trainImagesSmall.to_numpy()

            if isinstance(self.trainLabelsSmall, pd.DataFrame):
                print("ModelManager: Converting trainLabelsSmall to numpy array")
                self.trainLabelsSmall = self.trainLabelsSmall.to_numpy()

            # Initialize models            
            self.trainers.append(NeuralNetworkModel(input_shape, 'fcnn'))
            self.trainers.append(XGBoostModel(input_shape, 'xgboost'))
            self.trainers.append(RandomForestModel(input_shape, 'rf'))
            self.trainers.append(SVMModel(input_shape, 'svm'))
            self.trainers.append(NeuralNetworkModel(input_shape, 'cnn'))

    def train(self, useSmallDatasetToTrain=True, gridParams=default_grid_search_cv_params):
        print("ModelManager: train")
        if useSmallDatasetToTrain:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.trainImagesSmall, self.trainLabelsSmall, test_size=0.1, random_state=42, stratify=self.trainLabelsSmall)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.trainImagesLarge, self.trainLabelsLarge, test_size=0.1, random_state=42, stratify=self.trainLabelsLarge)

        # Normalize pixel values (0-255) to (0-1)
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        print(f"ModelManager: Shape of X_train after reshaping: {self.X_train.shape}")
        print(f"ModelManager: Shape of X_test after reshaping: {self.X_test.shape}")
        print(f"ModelManager: Shape of y_train after reshaping: {self.y_train.shape}")
        print(f"ModelManager: Shape of y_test after reshaping: {self.y_test.shape}")

        models_local = []

        for model in self.trainers:
            print(f"ModelManager: Training model: {model._name}")
            if model._name in gridParams:
                with mlflow.start_run(run_name=f"{model._name}_training", nested=True):
                    print(f"ModelManager: Training {model._name}")
                    mlflow.set_tag("model", model._name)
                    grid_search_cv_params = gridParams[model._name]
                    print(f"ModelManager: Training {model._name} with grid_search_cv_params: {grid_search_cv_params}")
                    if type(grid_search_cv_params) is dict:
                        n_jobs = grid_search_cv_params['n_jobs']
                        cv = grid_search_cv_params['cv']
                        param_grid = grid_search_cv_params['param']
                        print(f"ModelManager: Default Training {model._name} with param_grid: {param_grid}, n_jobs: {n_jobs}, cv: {cv}")
                    else:
                        n_jobs = grid_search_cv_params.n_jobs
                        cv = grid_search_cv_params.cv
                        param_grid = grid_search_cv_params.param
                        print(f"ModelManager: Customized Training {model._name} with param_grid: {param_grid}, n_jobs: {n_jobs}, cv: {cv}")
                    model.train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test, param_grid, n_jobs, cv)
                    if model._model is not None:
                        models_local.append(model)
                        score = model._score

                        if score > self.best_model_score:
                            self.best_model = model
                            self.best_model_test_accuracy = model._test_accuracy
                            self.best_model_score = model._score
                            self.best_model_recall = model._recall
                            self.best_model_params = model._best_params
                            self.best_model_test_log_loss = model._test_log_loss
                            self.best_model_roc_auc_score = model._roc_auc_score
                            self.best_model_cm_filename = model._cm_filename
                            self.best_model_cv_results_filename = model._cv_results_filename
        
        self.models = models_local
        self.logBestModelToMLflow()
        self.ensemble_train()

    def ensemble_train(self):
        print("ModelManager: ensemble_train")
        if self.best_model is None:
            raise Exception("No best model found. Train the models first.")
        
        if len(self.models) < 2:
            raise Exception("At least 2 models are required for ensemble training.")

        with mlflow.start_run(run_name="ensemble_training", nested=True):
            mlflow.set_tag("model", "ensemble_model")
            # Create ensemble model using the trained models
            self.ensemble_model = EnsembleModel(models=self.models)
            voting_classifier = self.ensemble_model.createEnsembleModel(voting='soft')  # Use 'hard' for majority voting
            print(f"Ensemle model using Voting classifier created successfully. Here are the named_estimators {voting_classifier.named_estimators}")
            # Train the ensemble model and evaluate
            evaluation_metrics = self.ensemble_model.train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test, param_grid=None, n_jobs=-1, cv=3)
            self.ensemble_model.train(self.X_train, self.y_train)
            print("Ensemble Model Evaluation Metrics:", evaluation_metrics)
            # Predict test the ensemble model
            predictions = self.ensemble_model.predict(self.X_test)
            print("Ensemble Model Predictions:", predictions)
            self.save_ensemble_model()
            # Log the ensemble model to MLflow
            self.ensemble_model.log_self_to_mlflow(evaluation_metrics, self.ensemble_model_filename)
            print("Ensemble Model logged in MLflow")

    def save_best_model(self):
        print(f"BaseModel: save_model - Saving the best trained model to a pickle file")        
        if self.best_model is None:
            raise Exception("No trained model found to save.")
        with open(self.best_model_filename, "wb") as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved as {self.best_model_filename}")
        return self.best_model_filename
    
    def load_best_model(self):
        print(f"ModelManager ({self.best_model_filename}): load_best_model - Loading a trained model from a pickle file")
        if not os.path.exists(self.best_model_filename):
            raise Exception(f"Model file {self.best_model_filename} not found!")
        with open(self.best_model_filename, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {self.best_model_filename}")
        return model
    
    def save_ensemble_model(self):
        if self.ensemble_model is None:
            raise Exception("No trained ensemble model found to save.")
        print(f"EnsembleModel: save_model - Saving the trained ensemble model to a pickle file")  
        with open(self.ensemble_model_filename, "wb") as f:
            pickle.dump(self.ensemble_model.ensemble_model, f)
        print(f"Model saved as {self.ensemble_model_filename}")
        return self.ensemble_model_filename

    def logBestModelToMLflow(self):
        """Logs the best model's metrics and parameters to MLflow."""
        if self.best_model is None:
            print("ModelManager: No best model found, skipping MLflow logging.")
            return
        # Saving the best model to a pickle file
        self.save_best_model()

        with mlflow.start_run(run_name=f"logging_best_of_all_model", nested=True, description="Logging the best of all models by comparing best score of GridSearchCV of each models"):
            mlflow.set_tag("model", "best_model")
            try:
                mlflow.log_param("best_model_name",self.best_model._name)
                mlflow.log_param("best_model_type", type(self.best_model._model).__name__)
            except Exception as e:
                print(f"ModelManager: Error logging model to MLflow: {e}")
            mlflow.log_params(self.best_model_params)
            mlflow.log_metric("best_model_score", self.best_model_score)
            mlflow.log_metric("best_model_test_accuracy", self.best_model_test_accuracy)
            mlflow.log_metric("best_model_recall", self.best_model_recall)
            mlflow.log_metric("best_model_test_log_loss", self.best_model_test_log_loss)
            mlflow.log_metric("best_model_roc_auc_score", self.best_model_roc_auc_score)
            mlflow.log_artifact(self.best_model_cm_filename)
            mlflow.log_artifact(self.best_model_cv_results_filename)
            mlflow.log_artifact(self.best_model_filename)
            
        print(f"ModelManager: Best model logged in MLflow: {type(self.best_model).__name__}")

    def predict(self, X, ensemble_predict = False):
        print("ModelManager: predict")
        if ensemble_predict is False:
            if self.best_model is None:
                if os.path.exists(self.best_model_filename):
                    best_model = self.load_best_model()
                    print("ModelManager: predict->Best model loaded from file")                   
                    best_model_name = None
                    if isinstance(best_model, KerasClassifier):
                        best_model_name = 'cnn' if self.is_cnn_model(best_model) else 'fcnn'
                    
                    if best_model_name == 'cnn':
                        print("ModelManager: predict->Reshaping input to 2D for CNN")
                        X = X.reshape(-1, 28, 28, 1)  # Reshape to (-1, 28, 28, 1)
                        print(f"ModelManager: predict->Shape of input X after reshaping: {X.shape}")
                    else:
                        X = X.reshape(-1, 28 * 28)  # Reshape to (-1, 784)
                    return best_model.predict(X)
                else:
                    raise Exception("Model not trained yet") 
            else:
                # Ensure input shape matches the best model's expectations
                if isinstance(self.best_model, NeuralNetworkModel) and self.best_model._name == 'cnn':
                    print("ModelManager: predict->Reshaping input to 2D for CNN")
                    X = X.reshape(-1, 28, 28, 1)  # Reshape to (-1, 28, 28, 1)
                    print(f"ModelManager: predict->Shape of input X after reshaping: {X.shape}")
                else:
                    X = X.reshape(-1, 28 * 28)  # Reshape to (-1, 784)
                return self.best_model.predict(X)
        else:
            print("ModelManager: predict->Ensemble prediction")
            if self.ensemble_model is None:
                if os.path.exists(self.ensemble_model_filename):                    
                    print("ModelManager: predict->Ensemble model is trained. Predicting using ensemble model")
                    X = X.reshape(-1, 28 * 28)  # Reshape to (-1, 784)
                    ensemble_model = EnsembleModel(self.models)
                    return ensemble_model.predict(X, self.ensemble_model_filename)
                else:
                    print("ModelManager: predict->Ensemble model is not trained yet")
                    raise Exception("Ensemble model not trained yet")
            else:
                print("ModelManager: predict->Ensemble model is trained. Predicting using ensemble model")
                X = X.reshape(-1, 28 * 28)  # Reshape to (-1, 784)
                return self.ensemble_model.predict(X, self.ensemble_model_filename)
            
    def is_cnn_model(self, keras_model):
        if keras_model.build_fn == build_cnn:
            return 'cnn'
        else:
            return 'fcnn'