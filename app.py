import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from flask import Flask,url_for,request,render_template,redirect
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename


tf.keras.backend.clear_session()

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Define a flask app
app = Flask(__name__)

# Type of Chest-xray cases that we are going to predict

xray_type = ['COVID19','NORMAL','PNEUMONIA']

# load the model from json file
json_file = open('./covid19_high_accuracy_classification_model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

# Load the weights into a model

model.load_weights('./covid19_high_accuracy_model_29-0.992030.h5')
print("Model Loaded Successfully")

#
model.summary()
model._make_predict_function()

def model_predict(image_path,model):
    img = image.load_img(image_path,target_size=(350,350))
    print(img)
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    # Rescaling the image
    x = x/255.0
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('./index.html')


@app.route('/',methods=['POST','GET'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,'uploads',secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path,model)
        print(preds)
        pred_class = np.argmax(preds)
        result = xray_type[pred_class]
        return render_template('./predict.html',result=result)
    else:
        return render_template('./index.html')

if __name__ == '__main__':
    app.run(debug=True)

