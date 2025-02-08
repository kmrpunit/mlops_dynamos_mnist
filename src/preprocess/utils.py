import os
import urllib.request
import gzip
import shutil

dataset_dir = "../dataset"

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
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    urls = {
        "train-images-idx3-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte" : "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    }
    
    for name, url in urls.items():
        file_path = f"{dataset_dir}/{name}"
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
        convert(f"{dataset_dir}/train-images-idx3-ubyte", f"{dataset_dir}/train-labels-idx1-ubyte",
            train_file, 60000)
    
    if not os.path.exists(test_file):
        print(f"Converting test data to {test_file}...")
        convert(f"{dataset_dir}/t10k-images-idx3-ubyte", f"{dataset_dir}/t10k-labels-idx1-ubyte",
            test_file, 10000)

if __name__ == "__main__":
    generate_mnist_csv(f"{dataset_dir}/mnist_train.csv", f"{dataset_dir}/mnist_test.csv")
