import librosa
import librosa.display
import argparse
import matplotlib.pyplot as plt
import numpy as np

def resample(file_path):
    '''
        Description: Loads audio, converts to mono, resamples to 12kHz, and 
        normalizes for loudness +1,-1
        Parameters: str file_path
        Returns:
            y: single channel audio time series, shape (n,)
            sr: sample rate of y, defaults to 12kHz
    '''
    y, sr = librosa.load(file_path, sr=12000, mono=True)
    return y, sr

def time_scale(y, time=5):
    if y.shape[0]<time*sr:
        y=np.pad(y,int(np.ceil((time*sr-y.shape[0])/2)),mode='reflect')
    else:
        y=y[:time*sr]
    return y

def get_melspectrogram_db(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    y = time_scale(y, 30) #creates interval of 30 seconds to compare across
    spec = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    return spec_db

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to audio file")
    args = parser.parse_args()
    y, sr = resample(args.file)
    spec_db = get_melspectrogram_db(y,sr)
    plt.show()
