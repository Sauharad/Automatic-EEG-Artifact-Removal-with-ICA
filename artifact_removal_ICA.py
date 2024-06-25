import mne
import numpy as np
import matplotlib.pyplot as plt

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
raw.filter(l_freq=1.0, h_freq=40.0)

orig = raw.copy()

#ICA Decomposition of 20 Components

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)

ica_comps = ica.get_sources(raw)
sources = ica_comps.get_data()

mixing_matrix = ica.mixing_matrix_

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

#Removing the Artifacted Component from ICA Reconstructiom

ica.exclude = artifact_components

#Applying ICA

ica.apply(raw)

#Visualising the Original and Denoised Signals

orig.plot()
raw.plot()

plt.show()
