import os
import urllib.request
import gzip
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

train_file = "./dataset/mnist_train.csv"
test_file = "./dataset/mnist_test.csv"

def convert(imgf, labelf, outf, n):
    with open(imgf, "rb") as f, open(labelf, "rb") as l, open(outf, "w") as o:
        f.read(16)
        l.read(8)
        images = []

        for i in range(n):
            image = [int.from_bytes(l.read(1), byteorder='big')]
            image.extend([int.from_bytes(f.read(1), byteorder='big') for _ in range(28*28)])
            images.append(image)

        for image in images:
            o.write(",".join(map(str, image)) + "\n")

def init_mnist():
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")

    urls = {
        "train-images-idx3-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    }

    for name, url in urls.items():
        file_path = f"./dataset/{name}"
        if not os.path.exists(file_path):
            gz_file_path = f"{file_path}.gz"
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, gz_file_path)
            print(f"Extracting {gz_file_path}...")
            with gzip.open(gz_file_path, "rb") as f_in, open(file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_file_path)  # Clean up the compressed file
            print(f"{file_path} ready.")

def generate_mnist_csv(train_file, test_file):
    init_mnist()
    if not os.path.exists(train_file):
        print(f"Converting training data to {train_file}...")
        convert("./dataset/train-images-idx3-ubyte", "./dataset/train-labels-idx1-ubyte",
            train_file, 60000)

    if not os.path.exists(test_file):
        print(f"Converting test data to {test_file}...")
        convert("./dataset/t10k-images-idx3-ubyte", "./dataset/t10k-labels-idx1-ubyte",
            test_file, 10000)

if __name__ == "__main__":
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        generate_mnist_csv("./dataset/mnist_train.csv", "./dataset/mnist_test.csv")



# Function to plot images
def plot_images(images, labels, num_images=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

class ReadAndLoadMnistData:

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_df = None
        self.test_df = None

    def __enter__(self):
        # Ensuring files are generated
        generate_mnist_csv(self.train_file, self.test_file)

        # Load the CSV files into DataFrames
        self.train_df = pd.read_csv(self.train_file)
        self.test_df = pd.read_csv(self.test_file)

        return self

    def load_data(self):
        # Separate features and labels for both train and test datasets
        X_train = self.train_df.iloc[:, 1:]
        y_train = self.train_df.iloc[:, 0]
        X_test = self.test_df.iloc[:, 1:]
        y_test = self.test_df.iloc[:, 0]
        return X_train, y_train, X_test, y_test

    def __exit__(self, exc_type, exc_value, traceback):
        # Clean up DataFrames
        del self.train_df
        del self.test_df

        if exc_type:
            print(f"Exception has been handled: {exc_value}")
        else:
            print("No exception occurred")

        # Return True to suppress any exception
        return True  


def save_best_model(model, accuracy, model_dir="model"):
    """
    Save the model if it has the best accuracy so far
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    best_accuracy_file = os.path.join(model_dir, "best_accuracy.txt")
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    
    # Check if there's an existing best accuracy
    current_best = 0
    if os.path.exists(best_accuracy_file):
        with open(best_accuracy_file, 'r') as f:
            current_best = float(f.read())
    
    # Save the model if it's better than the current best
    if accuracy > current_best:
        print(f"\nNew best model found! Accuracy improved from {current_best:.4f} to {accuracy:.4f}")
        # Save the model
        joblib.dump(model, best_model_path)
        # Save the accuracy
        with open(best_accuracy_file, 'w') as f:
            f.write(str(accuracy))
        print(f"Model saved to: {best_model_path}")
        return True
    else:
        print(f"\nCurrent model accuracy ({accuracy:.4f}) did not improve upon best accuracy ({current_best:.4f})")
        return False

def main():
    # Define file paths
    train_file = "./dataset/mnist_train.csv"
    test_file = "./dataset/mnist_test.csv"
    
    # Load and prepare data
    with ReadAndLoadMnistData(train_file, test_file) as mnist_data:
        # Get the data
        X_train, y_train, X_test, y_test = mnist_data.load_data()
        
        # Print shapes to verify data loading
        print("Data shapes:")
        print(f"Training data: {X_train.shape}, {y_train.shape}")
        print(f"Test data: {X_test.shape}, {y_test.shape}")
        
        # Plot some sample images
        print("Plotting sample training images...")
        plot_images(X_train.values, y_train.values)
        
        # Normalize and reshape data
        X_train = X_train.values / 255.0
        X_test = X_test.values / 255.0
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        
        # Train and evaluate model
        with mlflow.start_run():
            # Define model and parameters
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            param_grid = {
                'max_depth': [3, 6],  # Reduced for faster training
                'learning_rate': [0.1, 0.3],
                'n_estimators': [50, 100]
            }
            
            # Perform grid search
            print("Starting grid search...")
            grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=1)
            grid_search.fit(X_train_split, y_train_split)
            
            # Get best model and evaluate
            best_model = grid_search.best_estimator_
            print(f"\nBest Parameters: {grid_search.best_params_}")
            
            # Evaluate on validation set
            y_val_pred = best_model.predict(X_val_split)
            val_accuracy = accuracy_score(y_val_split, y_val_pred)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Evaluate on test set
            y_test_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Save the model if it's the best so far
            save_best_model(best_model, test_accuracy)
            
            # Log metrics
            mlflow.log_metric("validation_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.sklearn.log_model(best_model, "model")
            
            
                    # Plot predictions vs actual values
        def plot_predictions(images, actual_labels, predicted_labels):
            plt.figure(figsize=(10, 5))
            for i in range(10):
                plt.subplot(2, 5, i+1)
                plt.imshow(images[i].reshape(28, 28), cmap='gray')
                plt.title(f"True: {actual_labels[i]}, Pred: {predicted_labels[i]}")
                plt.axis('off')
            plt.show()

            # Plot predictions
            print("\nPlotting predictions vs actual values...")
            plot_predictions(X_test, y_test, y_test_pred)

if __name__ == "__main__":
    main()
