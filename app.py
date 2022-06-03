from tkinter import Y
from flask import Flask, render_template, request
import cv2
from PIL import Image
from flask import request,jsonify
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
app = Flask(__name__)
img_width, img_height = 256, 256
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
 
model = load_model('C:/Users/LENOVO/Desktop/Miety/Rice Report1 100 ephocs.h5')
model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy','Precision','Recall']
)
@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['POST'])
def upload_file2():
      photo = request.files['file']
      x = np.array(Image.open(photo))
     # pred = ['Bactrial Leaf Blight', 'Brown Spot','False Smut','Sheath Blight','Leaf Blast']
      pred = ['Bacterial Leaf Blight', 'Brown Spot','False Smut','Leaf Blast','Healthy','Random Image','Sheath Blight']
      x = cv2.resize(x, (256, 256))
      x = image.img_to_array(x)
      x = np.expand_dims(x, axis=0)
      images = np.vstack([x])
      classes = model.predict(images, batch_size=10)
      print(pred[classes.argmax()]) 
      y=pred[classes.argmax()] 
      print(classes)
      classes[0]
      x=max(100*classes[0])
      print(x)
      Result={"Predicted Result":str(y),"Matched Features":str(x)}
      return  jsonify(Result)
		
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)