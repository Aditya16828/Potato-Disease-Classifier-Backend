from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKETNAME = "model_bucket12345"
CLASSNAMES = ['Early blight', 'Late blight', 'Healthy']

model = None

def downloadBlob(bucketName, sourceBlob, destFile):
    storageClient = storage.Client()
    bucket = storageClient.get_bucket(bucketName)
    blob = bucket.blob(sourceBlob)
    blob.download_to_filename(destFile)

    print(f"Blob {sourceBlob} downloaded to {destFile}.")

def predict(request):
    global model
    if model is None:
        downloadBlob(BUCKETNAME, "models/potatoModel.h5", "/tmp/potatoModel.h5")
        model = tf.keras.models.load_model("/tmp/potatoModel.h5")
    
    img = request.files["file"]

    img = np.array(Image.open(img).convert("RGB").resize((256, 256)))
    img = img/255
    imgArr = tf.expand_dims(img, 0)
    
    predictions = model.predict(imgArr)
    print(predictions)

    predictedClass = CLASSNAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100

    return {"predictedClass": predictedClass, "confidence": confidence}