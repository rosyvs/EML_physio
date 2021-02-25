# EML_physio
physiological signal analysis (EDA, PPG) for EyeMindLink
EDA is the electrodermal activity measured with the electrodes on the palm. This can be decomposed into tonic (slow) changes and phasic (fast, stimulus-related) changes.

PPG can be processed to extract heart beats, and from that we can get both heart rate and heart rate variability.

The analyses will use [Neurokit2](https://neurokit2.readthedocs.io/en/latest/)

First steps: 
1. read in Shimmer exported .csvs for each subject. Extract GSR-Skin_Conductance and PPG columns along with their timestamps. 
2. Preprocess EDA: [nk.standardize](https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=nk.standardize#neurokit2.stats.standardize)
3. extract phasic and tonic components of EDA using [nk.ppg_process](https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=standardize#neurokit2.ppg.ppg_process)
4. Preprocess PPG: [nk.standardize](https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=nk.standardize#neurokit2.stats.standardize)
5. Extract HR from PPG: [nk.ppg_process](https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=standardize#neurokit2.ppg.ppg_process)
6. Extract HRV (heart rate variability) using [nk.hrv](https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=standardize#neurokit2.hrv.hrv)
7. Epoch the data: extract segemts which correspond to reading a page using [nk.epochs_create](https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=nk.standardize#neurokit2.epochs.epochs_create). This will require aligning the UNIX timestamps in the Shimmer data with the page start timestamps in the log files (pID_Trials.txt) and the page reading times from Dataviewer. 
