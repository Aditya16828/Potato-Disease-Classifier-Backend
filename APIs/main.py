from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from os import environ as env

endpoint = env.get('ENDPOINT')

CLASSNAMES = ['Early blight', 'Late blight', 'Healthy']

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "https://potato-disease-classifier.netlify.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# endpoint = "http://localhost:5000/v1/models/potato_disease_classifier_app:predict"
# endpoint = "https://us-central1-potato-disease-classification2.cloudfunctions.net/predict"

# Model = tf.keras.models.load_model('../Models/2')

@app.get('/')
async def index():
    return "Hello"


@app.post("/predict")
# async def predict(file: UploadFile):
#     image = np.array(Image.open(BytesIO(await file.read())))
#     image_batch = np.expand_dims(image, 0)
#     print(image_batch)
#     # prediction = Model.predict(np.array([image]))

#     json_data = {
#         "instances": image_batch.tolist()
#     }

#     response = requests.post(endpoint, json=json_data)
#     # return response.json()
#     prediction = np.array(response.json()["predictions"][0])
#     predictedClass = CLASSNAMES[np.argmax(prediction)]
#     predictionConf = np.max(prediction)*100
#     return {"predictedClass": predictedClass, "confidence": predictionConf}
async def predict(file: UploadFile):
    image = await file.read()
    data = {'file': ('image.jpg', image)}
    response = requests.post(endpoint, files=data)
    response = response.json()
    return response


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)