from pathlib import Path
import pandas as pd
import numpy as np
import os
from os import path
from zipfile import ZipFile
import pydub
from pydub import AudioSegment
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score,f1_score,precision_score
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import LabelEncoder

#Link of recorder
#https://github.com/Stephino/stephino.github.io/blob/master/tutorials/B3wWIsNHPk4/index.html

app = Flask(__name__)

def sortList(ListLengths):
    for x in range(0,len(ListLengths)):
        for y in range(0,len(ListLengths)-1):
            if(ListLengths[y][0] > ListLengths[y+1][0]):
                ListLengths[y],ListLengths[y+1] = ListLengths[y+1],ListLengths[y]
    return ListLengths

prediction = []
Models = []

@app.route('/')
def index():
    return render_template('recorder.html')

@app.route('/DetailedResults', methods=['GET', 'POST'])
def DetailedResults():
    return render_template('AllResults.html', list_html = Models)

@app.route('/Saleh/', methods=['GET', 'POST'])
def results():
    if request.method == 'GET':
        Path = "/home/saleh/Downloads"
        os.chdir(Path)
        
        FileRecording = "TestData.wav"

        X , SR = librosa.load(FileRecording)

        clip, index = librosa.effects.trim(X, top_db = 10)

        #MFCC Features
        MFCC = librosa.feature.mfcc(y = clip, sr = SR)
        MFCC = MFCC.flatten()
        
        #MFCC Transpose Features
        MFCC_T = (MFCC.T).flatten()
        
        #STFT features
        C = np.abs(librosa.stft(clip))
        chroma = librosa.feature.chroma_stft(y = C.flatten(), sr = SR)
        chroma = chroma.flatten()


        maxLength = 4644

        if(len(MFCC) < maxLength):
            MFCC = np.pad(MFCC,(0,maxLength-len(MFCC)))

        if(len(MFCC_T) < maxLength):
            MFCC_T = np.pad(MFCC_T,(0,maxLength-len(MFCC_T)))

        if(len(chroma) < maxLength):
            chroma = np.pad(chroma,(0,maxLength-len(chroma)))

        NewFeatures = np.concatenate((MFCC,chroma,MFCC_T))

        os.chdir("/home/saleh/Documents/NUCES/Sem 3/PAI/i20-0605")
        
        FileNames = ["i200605_BaggingClassifier.pkl","i200605_KNN.pkl","i200605_RandomForestClassifier.pkl","i200605_MLP.pkl"
                    ,"i200605_ExtraTreesClassifier.pkl","i200605_GradientBoostingClassifier.pkl","i200605_OneVsRestClassifier.pkl",
                    "i200605_DecisionTree.pkl"]

        #Pickle Load
        for x in range(0,len(FileNames)):
            TempModel = pickle.load(open(FileNames[x], 'rb'))
            Res = TempModel.predict([NewFeatures])
            Name = TempModel.__class__.__name__
            Label = Res[0]
            if(Res[0] == 0):
                Label = "A.C"
            elif(Res[0] == 1):
                Label = "Bulb-Light"
            elif(Res[0] == 2):
                Label = "Song-Gaana"
            elif(Res[0] == 3):
                Label = "T.V"

            Models.append([Name,Label])
            prediction.append(Res[0])

        maxFrequency = 0
        res = prediction[0]
        for i in prediction:
            freq = prediction.count(i)
            if freq > maxFrequency:
                maxFrequency = freq
                res = i

        Path = "/home/saleh/Downloads"
        os.chdir(Path)
        os.remove("TestData.wav")



        if(res == 0):
            string = "A.C"
        elif(res == 1):
            string = "Bulb-Light"
        elif(res == 2):
            string = "Song-Gana"
        elif(res == 3):
            string = "T.V"
            
        return render_template('Prediction.html' , String = string)

    else:
        return redirect(url_for('results'))

@app.errorhandler(404)
def not_found(error):
    return "<h1>404</h1>"


if __name__ == '_app_':
    app.debug = True
    app.run(debug = True,use_reloader=True)