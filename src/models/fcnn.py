import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix

from src.models.base import BaseModel

# Function to build model, required for KerasClassifier
def build_fcnn(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Input(shape=(784,)))
    model.add(Dense(units=128, activation='relu', kernel_initializer=init))
    model.add(Dropout(0.10))
    model.add(Dense(units=128, activation='relu', kernel_initializer=init))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='softmax', kernel_initializer=init))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=f'./model/fcnn_model_{optimizer}.png', show_shapes=True, show_layer_names=True)
    return model

# Function to create model, required for KerasClassifier
def build_cnn(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))    
    model.add(Flatten())    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='softmax'))    
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # plot_model(model, to_file=f'./model/cnn_model_{optimizer}.png', show_shapes=True, show_layer_names=True)
    return model

# class MlflowLogger(Callback):
#     def __init__(self):
#         super().__init__()
        
#     def on_epoch_end(self, epoch, logs=None):
#         if logs is not None:
#             print(f"Logging metrics to MLflow: {epoch} - {logs}")

class NeuralNetworkModel(BaseModel):
    
    def __init__(self, input_shape, name="fcnn"):
        super().__init__(name, input_shape)
        print(f"NeuralNetworkModel ({self._name}): __init__")
        
        self.default_param_grid = { 
            'batch_size': [128, 256, 384],
            'epochs': [10, 15], 
            'optimizer': ['adam', 'rmsprop']
        }

        # Create the KerasClassifier
        self.keras_model = KerasClassifier(model=build_fcnn if self._name == 'fcnn' else build_cnn, verbose=1, init='glorot_uniform')

    def train(self, X_train, y_train, param_grid, n_jobs, cv):
        print(f"NeuralNetworkModel ({self._name}): train with param_grid: {param_grid} and n_jobs: {n_jobs} and cv: {cv}")

        # Create GridSearchCV with error_score='raise'
        grid = GridSearchCV(estimator=self.keras_model, param_grid=param_grid, n_jobs=n_jobs, cv=cv, error_score='raise', verbose=3, return_train_score=True)

        # Fit the grid search
        grid_result = grid.fit(X_train, y_train)
        return grid_result
    
    def evaluate_model(self, grid_result, X_test, y_test):

        print(f"NeuralNetworkModel ({self._name}): evaluate_model")
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
        print(f"NeuralNetworkModel ({self._name}): train_and_evaluate")
        # Set default param_grid if not provided        
        param_grid = self.default_param_grid if param_grid is None else param_grid

        print(f"Shape of y_train after reshaping: {y_train.shape}")
        print(f"Shape of y_test after reshaping: {y_test.shape}")

        # Convert the input shape to 2D
        if self._name == 'cnn':
            print("Reshaping input to 2D for CNN")
            X_train = X_train.reshape(-1, 28, 28, 1)  # Reshape to (num_samples, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)    # Reshape to (num_samples, 28, 28, 1)
            print(f"Shape of X_train after reshaping: {X_train.shape}")
            print(f"Shape of X_test after reshaping: {X_test.shape}")
        
        # with mlflow.start_run(nested=True):
        grid_result = self.train(X_train, y_train, param_grid, n_jobs, cv)
        self.evaluate_model(grid_result, X_test, y_test)

        # Save and log the trained model
        self.save_model()

        # Log the model to MLflow
        self.log_model_to_mlflow()
    

    # def log_histories_to_mlflow(self, grid_result):
    #     try:
    #         print("Logging model histories to MLflow")
    #         history = grid_result.best_estimator_.model.history.history
    #         for epoch, metrics in enumerate(zip(history['accuracy'], history['loss'])):
    #             acc, loss = metrics
    #             mlflow.log_metric(f"{self._name}_accuracy", acc, step=epoch)
    #             mlflow.log_metric(f"{self._name}_loss", loss, step=epoch)
    #     except Exception as e:
    #         print(f"Error logging model histories to MLflow: {str(e)}")
    