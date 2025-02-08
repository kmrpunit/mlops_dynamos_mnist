# Dataset Download url:
https://github.com/cvdfoundation/mnist?tab=readme-ov-file

## Step 1: Download below files from above GitHub repository
    1. train-images-idx3-ubyte.gz 
    2. t10k-labels-idx1-ubyte.gz 
    3. t10k-images-idx3-ubyte.gz 
    4. train-labels-idx1-ubyte.gz

## Step 2: Extract them inside 'dataset' folder and name them as
    1. train-images-idx3-ubyte 
    2. t10k-labels-idx1-ubyte 
    3. t10k-images-idx3-ubyte 
    4. train-labels-idx1-ubyte

   And Run the generate_mnist_csv.py code to convert them into CSV file.

OR

## Step: Run below to code to auto download 
# Example usage

from src.preprocess.preprocess import ReadAndLoadMnistData

train_file = "../dataset/mnist_train.csv"
test_file = "../dataset/mnist_test.csv"

with ReadAndLoadMnistData(train_file, test_file) as mnist_data:
    X_train, y_train, X_test, y_test = mnist_data.load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_train.head())