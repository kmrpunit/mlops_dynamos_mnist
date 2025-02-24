import os
import pickle
import mlflow
import mlflow.sklearn
import mlflow.keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import keras
from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    def __init__(self, name, input_shape, num_classes = 10):
        print(f'BaseModel.__init__: {name}')
        np.random.seed(0)
        self._name = name
        self._num_classes = num_classes
        self.input_shape = input_shape
        self.initialize()

    def initialize(self):
        self._score = 0.0
        self._recall = 0.0
        self._test_accuracy = 0.0
        self._test_log_loss = 0.0
        self._roc_auc_score = 0.0        
        self._cm_filename = f"./model/confusion_matrix_{self._name}.png"
        self._cv_results_filename = f'./model/cv_results_{self._name}.csv'
        self._model_filename = f"./model/best_model_{self._name}.pkl"
        self._model = None
        self._best_params = None

    @abstractmethod
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, param_grid, n_jobs, cv):
        pass
  
    def one_hot_encode(self, y_train, y_test):
        y_train = keras.utils.to_categorical(y_train, self._num_classes)
        y_test = keras.utils.to_categorical(y_test, self._num_classes)
        return y_train, y_test

    def save_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for Model {self._name} on Mnist dataset")
        plt.savefig(self._cm_filename)
        plt.close()
        return self._cm_filename
    
    def summarize_result(self, grid_result):
        cv_results = grid_result.cv_results_        
        # Convert the cv_results_ dictionary into a DataFrame
        cv_results_df = pd.DataFrame(cv_results)
        # Save the DataFrame to a CSV file
        cv_results_df.to_csv(self._cv_results_filename, index=False)
        # Display the entire DataFrame
        print(cv_results_df)
        return self._cv_results_filename
    
    def save_model(self):
        print(f"BaseModel: save_model - Saving the trained model for {self._name} to a pickle file")        
        if self._model is None:
            raise Exception("No trained model found to save.")
        with open(self._model_filename, "wb") as f:
            pickle.dump(self._model, f)
        print(f"Model saved as {self._model_filename}")
        return self._model_filename
    
    def load_model(self):
        print(f"BaseModel ({self._name}): load_model - Loading a trained model from a pickle file")
        if not os.path.exists(self._model_filename):
            raise Exception(f"Model file {self._model_filename} not found!")
        with open(self._model_filename, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {self._model_filename}")
        return model
    
    def log_model_to_mlflow(self):
        # Log model as artifact              
        mlflow.log_artifact(self._model_filename)
        try :
            if self._name == 'cnn' or self._name == 'fcnn':
                mlflow.keras.log_model(self._model, f"best_model_keras_{self._name}")
            else:
                mlflow.sklearn.log_model(self._model, f"best_model_sklearn_{self._name}")
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")

        # Log best hyperparameters
        mlflow.log_params(self._best_params)
        mlflow.log_metric(f"{self._name}_best_score", self._score)

        # Log the evaluation metrics
        mlflow.log_metric(f"{self._name}_test_accuracy", self._test_accuracy)
        mlflow.log_metric(f"{self._name}_test_loss", self._test_log_loss)
        mlflow.log_metric(f"{self._name}_roc_auc_score", self._roc_auc_score)
        mlflow.log_metric(f"{self._name}_recall_score", self._recall)

        # Log confusion matrix as artifact
        mlflow.log_artifact(self._cm_filename)

        # Log cv_results csv file as artifact
        mlflow.log_artifact(self._cv_results_filename)

        print("MLflow logging complete.")

    def get_best_params(self):
        print(f"BaseModel ({self._name}): get_best_params")
        if self._best_params is None:
            raise Exception("Model not trained yet")
        return { 'name' : self._name, 
                'best_params' : self._best_params, 
                'accuracy' : self._score, 
                'test_accuracy' : self._test_accuracy,
                'test_loss' : self._test_log_loss,
                'roc_auc_score' : self._roc_auc_score,
                'recall' : self._recall }

    def predict(self, X):
        print(f"BaseModel ({self._name}): predict")
        if self._model is None:
            raise Exception("Model not trained yet")
        return self._model.predict(X)