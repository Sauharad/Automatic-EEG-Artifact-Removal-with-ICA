# Automatic-EEG-Artifact-Removal-with-ICA

An implementation of this paper for removal of ocular artifacts from EEG signals: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6956025/

ElectroEncephalogGraphy signals are the electrical signals of the brain, and can be used to analyze the resting state or task related behaviour of the brain. However, they are contaminated by artifacts created by eye blink, heatbeat and muscle movement artifacts. Of these, eye blinking artifacts (ElectroOculoGraphy - EOG) are the most prominent ones.

For the removal of these artifacts, the following approach has been employed in this code:

The EEG signal is decomposed into 20 components with Independent Component Analysis. Independent Component Analysis is a method of Blind Source Separation(BSS) which is able to decompose a mix of non gaussian signals into independent components with the condition of minimal mutual information. ICA finds the mixing and unmixing matrices which are used to form the signal from the components, and the components from the signals, respectively.

After obtaining the components, the Pearson Correlation of each component is calculated with the frontal EEG channels i.e. those channels closer to the eyes, which are far more likely to be artifacted. The components which have the maximum correlation are identified.

A threshold is calculated from the weights of the mixing matrix, and the correlated components which exceed this threshold are classifed as artifacts.

EOG peaks are detected within the component based on peak amplitude and time spread conditions, and 1 second epochs are formed around these peaks.
Wavelet decomposition is performed on these epochs, and then the epochs are reconstructed while retaining only the high frequency sub-bands, as the ocular artifacts are low frequency.

ICA reconstruction is performed from these filtered components, and the signal obtaining is free from ocular artifacts.
