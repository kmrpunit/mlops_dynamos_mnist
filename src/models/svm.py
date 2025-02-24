import numpy as np
from src.models.base import BaseModel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix


class SVMModel(BaseModel):    
    def __init__(self, input_shape, name='SVM'):
        super().__init__(name, input_shape)
        print("SVMModel: __init__")
        self.default_param_grid = {
                'C': [1, 10, 100],
                'gamma': [0.1, 0.01],
                'kernel': ['rbf']
            }
        self.svm_model = SVC(probability=True, random_state=42)
    
    def train(self, X_train, y_train, param_grid, n_jobs, cv):
        print(f"SVMModel ({self._name}): train with param_grid: {param_grid} and n_jobs: {n_jobs} and cv: {cv}")

        # Create GridSearchCV with error_score='raise'
        grid = GridSearchCV(estimator=self.svm_model, param_grid=param_grid, n_jobs=n_jobs, cv=cv, error_score='raise', verbose=3, return_train_score=True)

        # Fit the grid search
        grid_result = grid.fit(X_train, y_train)
        return grid_result
    
    def evaluate_model(self, grid_result, X_test, y_test):

        print(f"SVMModel ({self._name}): evaluate_model")
        self._model = grid_result.best_estimator_
        self._score = grid_result.best_score_
        self._best_params = grid_result.best_params_

        y_pred = self._model.predict(X_test)
        y_pred_proba = self._model.predict_proba(X_test)

        # Log evaluation metrics
        self._test_accuracy = accuracy_score(y_test, y_pred)
        self._test_log_loss = log_loss(y_test, y_pred_proba)
        self._roc_auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        self._recall = recall_score(y_test, y_pred, average='macro')

        print(f"Test Loss: {self._test_log_loss}, Test Accuracy: {self._test_accuracy}")
        print(f"ROC AUC Score: {self._roc_auc_score}")
        print(f"Recall Score: {self._recall}")

        # Summarize results 
        self.summarize_result(grid_result)

        # Save confusion matrix
        confusion_mtx = confusion_matrix(y_test, y_pred)
        self.save_confusion_matrix(confusion_mtx)
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, param_grid, n_jobs, cv):
        print("SVMModel: train_and_evaluate")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Shape of y_test: {y_test.shape}")

        # Perform grid search
        # Set default param_grid if not provided        
        param_grid = self.default_param_grid if param_grid is None else param_grid
        
        # with mlflow.start_run(nested=True):
        grid_result = self.train(X_train, y_train, param_grid, n_jobs, cv)
        self.evaluate_model(grid_result, X_test, y_test)

        # Save and log the trained model
        self.save_model()

        # Log the model to MLflow
        self.log_model_to_mlflow()