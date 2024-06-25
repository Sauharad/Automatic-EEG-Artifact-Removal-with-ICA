import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks
from pywt import wavedec, waverec

#Loading the Dataset
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file,preload=True)

#Picking the EEG channels and filtering

raw.pick(picks='eeg')
raw.drop_channels(ch_names=raw.info['bads'])
sfreq = raw.info['sfreq']
raw.crop(tmin=0,tmax=5)
raw.filter(l_freq=1.0, h_freq=40.0)

orig = raw.copy()

data = raw.get_data()

#ICA Decomposition of 20 Components

ica = FastICA(n_components=20, random_state=97, max_iter=800)
sources = ica.fit_transform(data.T)

sources = sources.T

mixing_matrix = ica.mixing_
print(mixing_matrix.shape)

#Selecting the Frontal EEG Channels

channels = [f'EEG 00{p}' for p in range(1,10)]
channels.extend([f'EEG 0{p}' for p in range(10,17)])

#Calculating the Pearson Correlation between the ICA Components and the Frontal Channels

correlation = []

for i in channels:
    eeg = raw.get_data(picks = i)
    this_component = []
    for j in range(0,len(sources)):
        cor = (np.cov(np.vstack([eeg,sources[j]])) / (np.std(eeg) * np.std(sources[j])))[0][1]
        this_component.append(cor)
    correlation.append(this_component)

#Finding the Components with the Highest Correlation to the Frontal Channels

eog_candidates = []

for i in correlation:
    j = np.argmax(np.abs(i))
    eog_candidates.append(j)

#Calculating the Weight Threshold to Classify a Component as EOG Artifact

weight_vector = {}

for j in eog_candidates:
    x = mixing_matrix[0:-1][j]
    y = [x[p] for p in range(0,len(channels))]
    wij = np.sum(np.abs(y))
    wij /= len(channels)
    weight_vector.update({j: wij})

threshold = np.percentile(list(weight_vector.values()),75) + 1.5 * (np.percentile(list(weight_vector.values()),75) - np.percentile(list(weight_vector.values()),25))

artifact_components = [j for j in weight_vector if weight_vector[j] >= threshold]
print(artifact_components)

#Localizing EOG peaks within the Artifacted Component

for i in artifact_components:
    #Localizing EOG peaks within the Artifacted Component.

    threshold = np.mean(sources[i])
    peaks = find_peaks(sources[i],height = threshold, distance = int(sfreq/2))
    peaks = peaks[0]
    for j in peaks:
        
        #Creating 1 second Epochs around the peaks

        print(j,j-int(sfreq/2),j+int(sfreq/2))
        if (j-int(sfreq/2)) <= 0:
            epoch = sources[i][0:j+int(sfreq/2)]
        else:
            epoch = sources[i][j-int(sfreq/2):j+int(sfreq/2)]

        # Reconstructing the independent component from only the high frequency bands of the wavelet decomposition
        coeffs = wavedec(epoch,'sym4',level=5)
        for k in range(0,len(coeffs)-2):
            coeffs[k] = np.zeros(coeffs[k].shape)
        cleaned_epoch = waverec(coeffs,'sym4')
        if (j-int(sfreq/2)) <= 0:
            sources[i][0:j+int(sfreq/2)+1] = cleaned_epoch
        else:
            sources[i][j-int(sfreq/2):j+int(sfreq/2)] = cleaned_epoch[0:len(sources[i][j-int(sfreq/2):j+int(sfreq/2)])]
 
#Reconstructing the Signal

filtered = np.dot(sources.T,mixing_matrix.T)
filtered = filtered.T

#Visualising the Original and Denoised Signals

new = mne.io.RawArray(filtered,raw.info)

orig.plot()

new.plot()

plt.show()
