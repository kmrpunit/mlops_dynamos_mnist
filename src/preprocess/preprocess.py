
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess.utils import generate_mnist_csv

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