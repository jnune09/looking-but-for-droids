import requests

KERAS_REST_API_URL = "http://localhost:5005/api/predict"
IMAGE_PATH = "./images/c3po.jpg"

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()

if r:
    print(r)
else:
    print("Request failed")
