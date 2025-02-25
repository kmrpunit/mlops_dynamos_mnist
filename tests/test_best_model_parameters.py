import requests

url = "http://127.0.0.1:8001/best_model_parameters"  # Replace with your actual API endpoint

response = requests.get(url)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Failed to get prediction. Status code:", response.status_code)