#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


#Setting Main Path
MainPath = "/home/saleh/Documents/NUCES/Sem 3/PAI/i20-0605"
os.chdir(MainPath)


# In[8]:


#Find te zip folder and extract
File = os.listdir()
chosen = ""
for x in File:
    if(x.find(".zip") > 0):
        chosen = x
        break

with ZipFile(chosen, 'r') as zipObj:
    zipObj.extractall()


# In[9]:


#Initializing Arrays
NPListLabel = np.empty(1)
ListMFCC = [[0]]
ListMFCCTranspose = [[0]]
ListChroma = [[0]]


# In[10]:


#Reconvert the audios to wav format
os.chdir(MainPath + "/ProjectAllfiles")
ListRecordingFolders = sorted(os.listdir())
for Labels in ListRecordingFolders:
    TempPath = MainPath + "/ProjectAllfiles/" + Labels
    os.chdir(TempPath)
    for Recording in os.listdir():
        InputFile = Recording
        OutPutFile = InputFile.replace("mp3","wav").replace("m4a","wav").replace("aac","wav").replace("mpeg","wav")
        sound = AudioSegment.from_file(InputFile)
        sound.export(OutPutFile, format = "wav")
    os.chdir("../")


# In[11]:


#Finding features from the audios
os.chdir(MainPath + "/ProjectAllfiles")
for Folder in sorted(os.listdir()):
    TempPath = MainPath + "/ProjectAllfiles"+ "/" + Folder
    os.chdir(TempPath)
    for recordingFile in os.listdir():
        NPListLabel = np.append(NPListLabel,Folder)
        
        X , SR = librosa.load(recordingFile,res_type='kaiser_fast')
        
        #Trimming the scilent parts of the audio from the ends
        Trim, index = librosa.effects.trim(X, top_db = 10)
        
        #MFCC Features
        MFCC = librosa.feature.mfcc(y = Trim, sr = SR)
        ListMFCC.append(MFCC.flatten())
        
        #MFCC Transpose Features
        MFCC_T = (MFCC.T)
        ListMFCCTranspose.append(MFCC_T.flatten())
        
        #STFT features
        C = np.abs(librosa.stft(Trim))
        chroma = librosa.feature.chroma_stft(y = C.flatten(), sr = SR)
        ListChroma.append(chroma.flatten())

    os.chdir("../")


# In[12]:


#Function to sort a list
def sortList(ListLengths):
    for x in range(0,len(ListLengths)):
        for y in range(0,len(ListLengths)-1):
            if(ListLengths[y][0] > ListLengths[y+1][0]):
                ListLengths[y],ListLengths[y+1] = ListLengths[y+1],ListLengths[y]
    return ListLengths


# In[13]:


DF = pd.DataFrame()

ListMFCC = ListMFCC[1:]
ListChroma = ListChroma[1:]
NPListLabel = NPListLabel[1:]
ListMFCCTranspose = ListMFCCTranspose[1:]

ListLengths = []
count = -1
for x in ListMFCC:
    count += 1
    ListLengths.append([len(x),count,x])

ListLengths = sortList(ListLengths)
    
ListDeleted = []
ListDeleted.append(ListLengths[0])
ListDeleted.append(ListLengths[len(ListLengths)-1])

ListLengths = ListLengths[1:len(ListLengths)-1]

#Label encoding labels column
le = LabelEncoder()
NEWListLabel = le.fit_transform(NPListLabel)

DF["Label"] = NEWListLabel

#Finding max length of MFCC features
maxLengthMFCC = 0
for x in ListLengths:
    if(x[0] > maxLengthMFCC):
        maxLengthMFCC = x[0]

#Finding max length of chroma features
maxLengthChroma = 0
for x in ListChroma:
    if(len(x) > maxLengthChroma):
        maxLengthChroma = len(x)

#Finding max length of Transpose of MFCC features
maxLengthMFCCTranspose = 0
for x in ListMFCCTranspose:
    if(len(x) > maxLengthMFCCTranspose):
        maxLengthMFCCTranspose = len(x)
        
maxLength = (max(maxLengthMFCC, maxLengthChroma, maxLengthMFCCTranspose))

#Padding all three feature lists
NewMFCCList = []
for x in ListMFCC:
    if(len(x) <= maxLength):
        NewMFCCList.append(np.pad(x,(0,maxLength-len(x))))
    else:
        NewMFCCList.append(x)

NewChromaList = []
for x in ListChroma:
    if(len(x) <= maxLength):
        NewChromaList.append(np.pad(x,(0,maxLength-len(x))))
    else:
        NewChromaList.append(x)

NewMFCCTransposeList = []
for x in ListMFCCTranspose:
    if(len(x) <= maxLength):
        NewMFCCTransposeList.append(np.pad(x,(0,maxLength-len(x))))
    else:
        NewMFCCTransposeList.append(x)

#Concatenating padded lists into 1
New = []
for x in range(0,len(NewMFCCList)):
    temp = np.concatenate((NewMFCCList[x],NewChromaList[x],NewMFCCTransposeList[x]))
    New.append(temp)

#Column of Features
DF["Features"] = New 

#Deleting (3) rows which had outlier MFCC features 
for x in ListDeleted:
    DF = DF.drop(x[1])


# In[14]:


#Data Frame
DF = DF.set_index(np.arange(len(DF)))
print(DF)


# In[15]:


os.chdir(MainPath)


# In[16]:


#Get training and testing data
X_train, X_test, y_train, y_test = train_test_split(DF["Features"], DF["Label"], test_size = 0.2,random_state = 40)


# In[17]:


#Converting to lists to avoid any error while training
x_train = list(X_train)
x_test = list(X_test)


# In[18]:


#Applying Standard Sclarer
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_sc = sc.transform(x_train)
x_test_sc = sc.transform(x_test)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
ModelKNeighbours = KNeighborsClassifier(n_neighbors = 3,metric= "braycurtis", p = 1 )
ModelKNeighbours.fit(x_train_sc, y_train)
pred = ModelKNeighbours.predict(x_test_sc)


# In[20]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred,average = 'micro')*100)
print("Precision :", precision_score(y_test, pred,average = 'micro')*100)
print("F1 Score :",f1_score(y_test, pred, average = 'micro')*100)


# In[21]:


FileName = 'i200605_KNN.pkl'
pickle.dump(ModelKNeighbours, open(FileName,'wb'))


# In[22]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy', splitter = "best")
clf = clf.fit(x_train_sc, y_train)
pred = clf.predict(x_test_sc)


# In[23]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred,average = 'micro')*100)
print("Precision :", precision_score(y_test, pred,average = 'micro')*100)
print("F1 Score :",f1_score(y_test, pred, average='micro')*100)


# In[24]:


FileName = 'i200605_DecisionTree.pkl'
pickle.dump(clf, open(FileName,'wb'))


# In[25]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter = 500)
mlp.fit(x_train_sc , y_train)
pred = mlp.predict(x_test_sc)


# In[26]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred, average='macro')*100)
print("Precision :", precision_score(y_test, pred, average='macro')*100)
print("F1 Score :",f1_score(y_test, pred, average='macro')*100)


# In[27]:


FileName = 'i200605_MLP.pkl'
pickle.dump(mlp, open(FileName,'wb'))


# In[28]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(x_train_sc , y_train)
pred = RFC.predict(x_test_sc)


# In[29]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred, average='macro')*100)
print("Precision :", precision_score(y_test, pred, average='macro')*100)
print("F1 Score :",f1_score(y_test, pred, average='macro')*100)


# In[30]:


FileName = 'i200605_RandomForestClassifier.pkl'
pickle.dump(RFC, open(FileName,'wb'))


# In[31]:


from sklearn.ensemble import ExtraTreesClassifier
ETC = ExtraTreesClassifier()
ETC.fit(x_train_sc , y_train)
pred = ETC.predict(x_test_sc)


# In[32]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred, average='macro')*100)
print("Precision :", precision_score(y_test, pred, average='macro')*100)
print("F1 Score :",f1_score(y_test, pred, average='macro')*100)


# In[33]:


FileName = 'i200605_ExtraTreesClassifier.pkl'
pickle.dump(ETC, open(FileName,'wb'))


# In[34]:


from sklearn.ensemble import BaggingClassifier
GBC = BaggingClassifier()
GBC.fit(x_train_sc , y_train)
pred = GBC.predict(x_test_sc)


# In[35]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred, average='macro')*100)
print("Precision :", precision_score(y_test, pred, average='macro')*100)
print("F1 Score :",f1_score(y_test, pred, average='macro')*100)


# In[36]:


FileName = 'i200605_BaggingClassifier.pkl'
pickle.dump(GBC, open(FileName,'wb'))


# In[37]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
ORC = OneVsRestClassifier(SVC(), n_jobs = 4)
ORC.fit(x_train_sc , y_train)
pred = ORC.predict(x_test_sc)


# In[38]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred, average='macro')*100)
print("Precision :", precision_score(y_test, pred, average='micro')*100)
print("F1 Score :",f1_score(y_test, pred, average='micro')*100)


# In[39]:


FileName = 'i200605_OneVsRestClassifier.pkl'
pickle.dump(ORC, open(FileName,'wb'))


# In[40]:


from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(n_estimators = 15,learning_rate = .3)
GBC.fit(x_train_sc , y_train)
pred = GBC.predict(x_test_sc)


# In[41]:


print(confusion_matrix(y_test,pred))
print("Accuracy :" , accuracy_score(y_test,pred)*100)  
print("Recall :", recall_score(y_test, pred, average='macro')*100)
print("Precision :", precision_score(y_test, pred, average='macro')*100)
print("F1 Score :",f1_score(y_test, pred, average='macro')*100)


# In[42]:


FileName = 'i200605_GradientBoostingClassifier.pkl'
pickle.dump(GBC, open(FileName,'wb'))

