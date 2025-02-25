import requests
import json

url = "http://127.0.0.1:8001/training"  # Replace with your actual API endpoint

post_data = {
    "use_small_dataset": True,
    "grid_search_params": {
        'fcnn': { 
                # Hyperparameters for FCNN
                'param' : { 
                'batch_size': [32, 64, 128],
                'epochs': [5, 10],
                'optimizer': ['adam', 'rmsprop']
                }, 
                'n_jobs' : -1, 
                'cv' : 3
            },
        'svm': { 
                # Hyperparameters for SVM
                'param' : {
                    'C': [1, 10, 100],
                    'gamma': [0.1, 0.01],
                    'kernel': ['rbf']
                }, 
                'n_jobs' : -1, 
                'cv' : 3
            },
        'rf': { 
                # Hyperparameters for Random Forest
                'param' : { 
                'n_estimators': [50, 75, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
                }, 
                'n_jobs' : -1,
                'cv' : 3
            },
        'xgboost': { 
                # Hyperparameters for XGBoost                
                'param' : {
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.3],
                    'n_estimators': [50, 100]
                }, 
                'n_jobs' : -1, 
                'cv' : 3
            },
        'cnn': {
            # Hyperparameters for CNN
            'param': {
                'batch_size': [32, 64, 128],
                'epochs': [5, 10],
                'optimizer': ['adam', 'rmsprop']
            },
            'n_jobs': -1,
            'cv': 3
        }
    }
}

#json_data = json.dumps(post_data)
#print(json.dumps(post_data, indent=4))
response = requests.post(url, json=post_data)

if response.status_code == 200:
    print("Training:", response.json())
else:
    print("Failed to get Training. Status code:", response.status_code)