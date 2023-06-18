import os
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import librosa 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

Ravdess = "C:/Users/prash/Downloads/speech-emotion-recognition-ravdess-data"
ravdess_directory_list = os.listdir(Ravdess)
ravdess_directory_list.sort()

emotions = []
X = []
for dir in ravdess_directory_list:
    actor_dir = os.listdir(Ravdess + f"/{dir}")
    for file in actor_dir:
        components = (file.split(".")[0]).split("-")
        
        emotion = components[2]
        emotions.append(emotion)
        
        path = Ravdess + f"/{dir}" + f"/{file}"
        X.append(path)


encoder = LabelEncoder()
Y = encoder.fit_transform(emotions)
x_train,x_test,y_train,y_test = train_test_split(X, Y, train_size = 0.8, random_state = 42)

class Features():
    
    def augment_input(self, file_path, augmented_type = str()):
        data, sampling_rate = librosa.load(file_path)
        self.sampling_rate = sampling_rate
        
        if augmented_type == "noise":
            noise_amp = 0.035*np.random.uniform()*np.amax(data)
            data = data + noise_amp*np.random.normal(size=data.shape[0])
            return data

        if augmented_type == "stretch":
            return librosa.effects.time_stretch(data, rate = 0.8)

        if augmented_type == "shift":
            shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
            return np.roll(data, shift_range)

        if augmented_type == "pitch":
            return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor = 0.7)
        return data
    
    
    def extract_features(self, X):
        result = np.array([])
        
        #MFCC
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=self.sampling_rate, n_mfcc = 40).T, axis = 0)
        result=np.hstack((result, mfccs))

        #Chroma_STFT
        stft=np.abs(librosa.stft(X))
        chroma=np.mean(librosa.feature.chroma_stft(S = stft, sr = self.sampling_rate,n_chroma = 32).T,axis = 0)
        result=np.hstack((result, chroma))

        # #MEL
        mel=np.mean(librosa.feature.melspectrogram(y = X, sr = self.sampling_rate,n_mels = 128, fmax = 8000).T,axis = 0)
        result=np.hstack((result, mel))

        #ZCR
        Z = np.mean(librosa.feature.zero_crossing_rate(y = X),axis = 1)
        result=np.hstack((result, Z))

        #Root Mean Square
        rms = np.mean(librosa.feature.rms(y = X).T, axis = 0)
        result = np.hstack((result, rms))
    
        return result
    
    def get_feature_set(self, paths, augmented_type = str()):
        self.paths = paths
        self.augmented_type = augmented_type
        output_df = []

        for file in self.paths:
            X = self.augment_input(file, augmented_type)
            final_features = self.extract_features(X)
            output_df.append(final_features)
        return output_df
    

class AudioDataset(Dataset):
    def __init__(self, data, labels):
        super(AudioDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


    
ft = Features()
unaugmented_train_df = ft.get_feature_set(x_train)
unaugmented_test_df = ft.get_feature_set(x_test)

# Create the data loaders
train_dataset = AudioDataset(unaugmented_train_df,y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = AudioDataset(unaugmented_test_df,y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize the model
from model import SCNNB
model = SCNNB()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Training Done")
        # Evaluate the model on test dataset
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, labels in test_loader:
    #             outputs = model(inputs.float())
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     accuracy = correct / total
    # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Test Accuracy: {accuracy}")

train(model, train_loader, criterion, optimizer, 100)
torch.save(model.state_dict(), "ravdess_trained.pth")