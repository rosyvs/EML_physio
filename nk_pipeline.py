import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(data_fpath):

    # Read datafile.
    data = pd.read_csv(data_fpath, header=1, sep="\t", low_memory=False)
    # Remove first col.
    data = data.drop(0, axis=0)
    # Convert To Float data (pandas pulls in as strings because of the label col removed above).
    for col in data.columns:
        data[f"{col}"] = data[f"{col}"].astype("float")
    return data


def main(data_fpath="EML1_001_Session1_Shimmer_92EE_Calibrated_SD.csv"):

    data = load_data(data_fpath)
    fname = data_fpath[data_fpath.rfind("/")+1:+8]

    ## EDA: Electrodermal activity
    eda_raw = nk.standardize(data["Shimmer_92EE_GSR_Skin_Conductance_CAL"])
    eda, info = nk.eda_process(eda_raw,sampling_rate=51.2)
    eda.to_csv(f"../../Data/phys_processed/{fname}_EDA.csv")

    eda_fig = nk.eda_plot(eda)

    ## PPG: Photoplethysmography 
    ppg_raw = nk.standardize(data["Shimmer_92EE_PPG_A13_CAL"],sampling_rate=51.2)
    ppg, info = nk.ppg_process(ppg_raw, sampling_rate=51.2)
    ppg_raw.plot(label="raw PPG") 
    ppg_fig = nk.ppg_plot(ppg) 
    ppg_fig 
    ppg.to_csv(f"../../Data/phys_processed/{fname}_PPG.csv")
    
    
    
    plt.show() # Matplotlib plots should be showed at the end of the script because the call 
    #is blocking i.e. won't resume running until you close the plot

if __name__ == "__main__":
    main()
