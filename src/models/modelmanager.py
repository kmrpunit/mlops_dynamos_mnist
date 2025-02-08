from src.models.fcnn import FCNNTrainAndPredict
from src.models.rf import RandomForestTrainAndPredict
from src.models.svm import SVMTrainAndPredict
from src.models.xgboost import XGBoostTrainAndPredict
from src.preprocess.preprocess import ReadAndLoadMnistData

from sklearn.metrics import accuracy_score, precision_score, recall_score

dataset_dir = "./dataset"
train_file = f"{dataset_dir}/mnist_train.csv"
test_file = f"{dataset_dir}/mnist_test.csv"
input_shape = 28 * 28

class ModelManager:

    def __init__(self):
        print("ModelManager: __init__")
        self.trainers = []
        self.models = []
        
        with ReadAndLoadMnistData(train_file, test_file) as minst_data:
            self.X_train, self.y_train, self.X_test, self.y_test = minst_data.load_data()            
            # Normalize the images
            self.X_train = self.X_train.values / 255.0
            self.X_test = self.X_test.values / 255.0

            # Flatten the images
            self.X_train = self.X_train.reshape(-1, 28*28)
            self.X_test = self.X_test.reshape(-1, 28*28)       

            # initialize models
            self.trainers.append(FCNNTrainAndPredict(input_shape, self.X_train, self.y_train, self.X_test, self.y_test))
            self.trainers.append(RandomForestTrainAndPredict(input_shape, self.X_train, self.y_train, self.X_test, self.y_test))
            self.trainers.append(SVMTrainAndPredict(input_shape, self.X_train, self.y_train, self.X_test, self.y_test))
            self.trainers.append(XGBoostTrainAndPredict(input_shape, self.X_train, self.y_train, self.X_test, self.y_test))            

            # initialize best model and metrics
            self.best_model = None
            self.best_accuracy = 0.0
            self.best_score = 0.0
            self.best_recall = 0.0


    def train(self):
        print("ModelManager: train")
        for t in self.trainers:
            t.train_and_evaluate()

            # TODO: Update code to find better model
            if t.model is not None:
                self.models.append(t.model)
                y_pred = t.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                score = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                if accuracy > self.best_accuracy:                    
                    self.best_model = t.model
                    self.best_accuracy = accuracy
                    self.best_score = score
                    self.best_recall = recall

    def predict(self, X):
        print("ModelManager: predict")
        if self.best_model is None:
            raise Exception("Model not trained yet")
        return self.best_model.predict(X)