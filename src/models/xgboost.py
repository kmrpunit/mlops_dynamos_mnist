
class XGBoostTrainAndPredict:    
    def __init__(self, input_shape, X_train, y_train, X_test, y_test):
        print("XGBoostTrainAndPredict: __init__")
        self.input_shape = input_shape
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        self.best_accuracy = 0.0
        self.best_score = 0.0
        self.best_recall = 0.0
        self.model = None
        self.best_params = None    
    
    def train_and_evaluate(self):
        print("XGBoostTrainAndPredict: train_and_evaluate")
        # TODO: Train the model

    def get_best_params(self):
        print("XGBoostTrainAndPredict: get_best_params")
        if self.best_params is None:
            raise Exception("Model not trained yet")
        return self.best_params
    
    def predict(self, X):
        print("XGBoostTrainAndPredict: predict")
        if self.model is None:
            raise Exception("Model not trained yet")
        return self.model.predict(X)