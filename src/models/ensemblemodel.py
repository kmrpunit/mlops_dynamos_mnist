import os
import pickle
import mlflow
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.models.base import BaseModel

class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, shape):
        print("ReshapeTransformer __init__{}".format(shape))
        self.shape = shape

    def fit(self, X, y=None):
        print("ReshapeTransformer fit")
        return self

    def transform(self, X):
        print("ReshapeTransformer transform")
        return X.reshape(self.shape)

class EnsembleModel(BaseModel):
    def __init__(self, models):
        print("EnsembleModel __init__")
        super().__init__(name="ensemble", input_shape=None)
        self.models = models
        self.ensemble_model = None

    def createEnsembleModel(self, voting='soft'):
        estimators = []
        
        for model in self.models:
            print(f"createEnsembleModel Model: {model._name}, Type: {type(model._model)}")
            
            if model._name == 'cnn':
                pipeline = Pipeline([
                    ('reshape', ReshapeTransformer((-1, 28, 28, 1))),
                    ('model', model._model)
                ])
            else:
                pipeline = Pipeline([
                    ('reshape', ReshapeTransformer((-1, 28 * 28))),
                    ('model', model._model)
                ])

            estimators.append((model._name, pipeline))
        
        self.ensemble_model = VotingClassifier(estimators=estimators, voting=voting)
        return self.ensemble_model

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, param_grid, n_jobs, cv):
        print(f"EnsembleModel ({self._name}): train_and_evaluate")
        print(f"EnsembleModel Shape of y_train : {y_train.shape}")
        print(f"EnsembleModel Shape of y_test : {y_test.shape}")
        self.train(X_train, y_train)
        evaluation_metrics = self.evaluate(X_test, y_test)
        return evaluation_metrics

    def train(self, X_train, y_train):
        print(f"EnsembleModel ({self._name}): train")
        if self.ensemble_model is None:
            raise Exception("Ensemble model is not created yet. Call createEnsembleModel() first.")

        # Fit the ensemble model
        self.ensemble_model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):

        print(f"EnsembleModel ({self._name}): evaluate")

        if self.ensemble_model is None:
            raise Exception("Ensemble model is not trained yet. Call train() first.")

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        confusion_mtx = confusion_matrix(y_test, y_pred)
        self.save_confusion_matrix(confusion_mtx)

        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'test_predictions_shape': y_pred.shape,
            f'confusion_matrix_{self._name}': self._cm_filename
        }
    
    def load_ensemble_model(self, ensemble_model_filename):
        print(f"EnsembleModel ({ensemble_model_filename}): load_ensemble_model - Loading a trained model from a pickle file")
        if not os.path.exists(self.ensemble_model_filename):
            raise Exception(f"Model file {ensemble_model_filename} not found!")
        with open(ensemble_model_filename, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {self.ensemble_model_filename}")
        return model
    
    def predict(self, X, ensemble_model_filename = None):
        print(f"EnsembleModel ({self._name}): predict")
        if self.ensemble_model is None:
            if ensemble_model_filename is not None and os.path.exists(ensemble_model_filename):
                self.ensemble_model = self.load_ensemble_model(ensemble_model_filename)
            else:
                raise Exception("Ensemble model is not trained yet. Call train_and_evaluate() first.")

        if self.ensemble_model.voting == 'soft':
            avg_probs = self.ensemble_model.predict_proba(X)
            return np.argmax(avg_probs, axis=1)
        else:
            return self.ensemble_model.predict(X)

    def log_self_to_mlflow(self, evaluation_metrics, ensemble_model_filename):
        print("Logging Ensemble model to MLflow")        
        mlflow.log_params(evaluation_metrics)
        mlflow.log_artifact(self._cm_filename)
        mlflow.log_metric(f"{self._name}_accuracy", evaluation_metrics['accuracy'])
        mlflow.log_metric(f"{self._name}_recall", evaluation_metrics['recall'])
        mlflow.log_metric(f"{self._name}_precision", evaluation_metrics['precision'])
        mlflow.log_metric(f"{self._name}_f1_score", evaluation_metrics['f1_score'])        
        try:
            mlflow.log_artifact(ensemble_model_filename)
            mlflow.sklearn.log_model(self.ensemble_model, f"model_sklearn_{self._name}")
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
        print("MLflow Ensemble model logging complete.")