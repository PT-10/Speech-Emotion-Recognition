import io
import json
import torch
from model import SCNNB
from flask import *
from flask import Flask, request
import librosa
import numpy as np

app = Flask(__name__)
   
train_model = SCNNB()
train_model.load_state_dict(torch.load("ravdess_trained.pth"))

class Features():
    
    def augment_input(self, data, sampling_rate, augmented_type = str()):
        return data
    
    def extract_features(self, X, sampling_rate):
        result = np.array([])
        
        #MFCC
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sampling_rate, n_mfcc = 40).T, axis = 0)
        result=np.hstack((result, mfccs))

        #Chroma_STFT
        stft=np.abs(librosa.stft(X))
        chroma=np.mean(librosa.feature.chroma_stft(S = stft, sr = sampling_rate,n_chroma = 32).T,axis = 0)
        result=np.hstack((result, chroma))

        #MEL
        mel=np.mean(librosa.feature.melspectrogram(y = X, sr = sampling_rate,n_mels = 128, fmax = 8000).T,axis = 0)
        result=np.hstack((result, mel))

        #ZCR
        Z = np.mean(librosa.feature.zero_crossing_rate(y = X),axis = 1)
        result=np.hstack((result, Z))

        #Root Mean Square
        rms = np.mean(librosa.feature.rms(y = X).T, axis = 0)
        result = np.hstack((result, rms))
    
        return result

class_labels = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}

def get_prediction(file_io, sampling_rate):
    ft = Features()
    augmented_input = ft.augment_input(file_io, sampling_rate)
    input = ft.extract_features(augmented_input, sampling_rate)

    # Convert the features to a tensor and unsqueeze to add a batch dimension
    input_tensor = torch.tensor(input).unsqueeze(0)

    # Make a prediction using the train_model
    train_model.eval()
    with torch.no_grad():
        output = train_model(input_tensor.float())
        _, predicted = torch.max(output.data, 1)
        class_label = predicted.item()
        
    return class_labels[class_label]

   
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods = ['POST', 'GET'])  
def predict():
    if request.method == 'POST':
        file = request.files['file']
        with io.BytesIO(file.read()) as file_obj:
            data, sampling_rate = librosa.load(file_obj, sr=None)
        name = file.filename
        if file:
            prediction = get_prediction(data, sampling_rate)
            return f'File {name} uploaded successfully, predicted emotion is {prediction}'
        else:
            return "Please select a file"
    return f'Invalid xyz' 
  
   
if __name__ == '__main__':  
   app.run(debug = True)  