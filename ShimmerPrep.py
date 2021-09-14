import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv, os, datetime, time, pickle
from scipy.signal import correlate
from tqdm import tqdm
from datetime import datetime
from os import path



class Pipeline:
	def __init__(self):
		print("\nShimmer Pipeline Initializing\nProcess PID - " + str(os.getpid()))
		self.subject_pool = range(95, 96)
		self.sr = 51.2
		self.motion_threshold = 0.15
		self.max_motion_artifact = 1.0
		self.texts = ['CausalClaims', 'Validity', 'Variables', 'Bias', 'Hypotheses']
		self.pages = range(10)
		self.folder_prefix = '../../Dropbox (Emotive Computing)/EyeMindLink/Data/' # File path to EyeMindLink Data folder
		self.folder_suffix = '/Shimmer/'
		self.file_prefix = ''  # Shimmer file prefix to the subject ID
		self.file_suffix = '_Session1_Shimmer_92EE_Calibrated_SD.csv' # Shimmer file suffix to the subject ID
		self.output_folder = 'Processed_Shimmer/'
		self.plot_path = 'Matplotlib-Plots/' # Output path for all plots in the shimmer output director
		self.event_timestamps = {}
		self.summaries = {}
		self.failed = []
		self.success = []
		self.warning = []
		print('Pipeline Initialized :D')

	def run(self):
		self.failed = [] # Declare list of subjects who failed to be processed
		self.success = [] # Declare list of subjects who successfully were processed
		for self.subject in tqdm(self.subject_pool): # Iterate through subject pool
			self.process(self.subject, reprocess = True) # Call for subject to be processed
		print('Pipeline run complete!')
		self.status() # Call for an end summary to be printed

	def status(self):
		if len(self.success) > 0: # If there were successfully run subjects
			print('Here are the subjects successfully processed:\n •','\n • '.join(self.success), '\n') # print out list of successfully processed
		if len(self.failed) > 0: # If there were subjects who were failed to be processed
			print('Some Subjects Not Processed:\n •', '\n • '.join(self.failed), '\n') # print out list of subjects who failed to be processed
		if len(self.warning) > 0:
			print('Some warnings popped up while processing subjects:\n •', '\n • '.join(self.warning), '\n')


	def process(self, subject, reprocess = False, motion_correct = False):
		#-------------------------- Main Processing ----------------------#
		# This function will take the given parameters set during initialization,
		# to attempt to process a given subjects shimmer data (Note that you pass
		# in the subject number and their ID will be constructed). The function
		# will process the data by loading in all data if available, label the
		# data with events of interest, detect for motion artifacts (motion
		# correcting if passed as True in the function call) and dropping any
		# samples that aren't of interest before summarizing the data before
		# saving the data.
		#------------------------------------------------------------------#
		self.subject_ID = 'EML1_' + '0'*(3 - len(str(subject))) + str(subject) # Create the subject ID
		self.output_path = self.folder_prefix + self.subject_ID + self.folder_suffix # Declare the output path

		# Check if file already exists and cancel processing if reprocess was not passed as True
		if path.exists(self.output_path + self.subject_ID + '_labels.npy') == True and reprocess == False: # If it exists and reprocessing not allowed
			self.failed.append(self.subject_ID + ' previously processed, need reprocess set as true to process')
			return # halt processing of this subject

		# Load shimmer csv file
		filename = self.file_prefix + self.subject_ID + self.file_suffix # Declare shimmer filename
		if path.exists(self.output_path + filename) == False: # If shimmer file isn't found cancel processing
			self.failed.append(self.subject_ID + ' Shimmer file not found')
			return # Skip processing of this subject
		self.data = self.read_csv(self.output_path, filename, '\t') # Read in shimmer data
		if len(self.data) == 0:
			self.failed.append(self.subject + ' Shimmer file unable to be loaded')
			return
		if self.data[2][0] == 'Ticks': # If the shimmer file used ticks instead of
			self.failed.append(self.subject_ID + ' Shimmer file uses Ticks, unable to find events of interest')
			return # skip processing of this subject
		self.data = np.delete(self.data, [0, 1, 2], axis = 0) # Delete data type and unit rows
		self.data_timestamps = [float(datum[0])/1000 for datum in self.data] # Collect data timestamps

		# 3 - Load subjects events data
		filename = self.subject_ID + '_events.csv' # Declare event filename
		if path.exists(self.folder_prefix + self.subject_ID + '/' + filename) == False:
			self.failed.append(self.subject_ID + ' events file not found')
			return # skip processing of this subject
		self.events = self.read_csv(self.folder_prefix + self.subject_ID + '/', filename, ',') # Load events
		self.exp_start = float(self.events[1][26]) # Collect experiment start for labeling
		self.event_timestamps = [[float(event[26]), float(event[27]), event[19], event[20], event[21]] for event in self.events[1:] if event[18] == 'reading'] # Read in events that have a timestamp
		if len(self.event_timestamps) == 0: # Check if  the events were properly processed
			self.failed.append(self.subject_ID + ' unable to process events file')
			return # Skip processing of subject

		# === General shimmer data labels === #
		# This section iterates through the data unix timestamps and labels data
		# unless the data was collected before the experiment started or if
		# no event of interest was found then it labels it as -1. It grabs the
		# Text, Page and Shimmer labels from the event file currently.
		self.labels = []
		for timestamp in self.data_timestamps:
			if timestamp < self.exp_start: # If the data was collected before the experiment started
				self.labels.append([-1, -1, -1]) # Append -1's as the label
			else: # If the data was collected after the experiment started
				label = [[event[2], event[3], event[4]] for event in self.event_timestamps if event[0] < timestamp and timestamp <= event[1]]
				if len(label) == 1: # If a label was found
					self.labels.append(label[0]) # Append the label we found
				else:
					self.labels.append([-1, -1, -1]) # Append -1's as label

		# Find and remove unneeded samples
		self.unneeded = self.search(self.labels, [[-1, -1, -1]]) # Search for all labels that were not in range of events of interest or unlabeled

		# Preprocess EDA data using neurokit2 eda processing library
		# Visit https://neurokit2.readthedocs.io/en/latest/functions.html
		# for more information about the functions being called
		eda_raw = [float(signal[12]) for signal in self.data] #Convert EDA data strings to floats
		self.eda_signals, self.eda_info = nk.eda_process(eda_raw, sampling_rate=self.sr) #  Run a simple preprocessing of the data
		self.eda_data = nk.eda_phasic(nk.standardize(self.eda_signals["EDA_Clean"]), sampling_rate=self.sr) # Seperate signal into phasic and tonic data

		self.eda_signals['EDA_Artifacts'] = self.detect_motion_artifacts(self.eda_signals['EDA_Clean']) # Fancy calculus mafic?
		# === Potential Motion Correction === #
		# Summarize the artifact data to find the mean and standard deviations.
		# Then define a motion threshold using those mentrics that aren't already
		# set to be dropped and add them onto the unneeded list
		self.artifact_summary = self.summarize(self.eda_signals['EDA_Artifacts']) # Get Summary statistics of EDA artifacts
		if motion_correct == True: # Potential section to use for motion correction
			self.motion_threshold = self.artifact_summary[0] + 2*self.artifact_summary[1] # Determine motion threshold by the subjects average plus two standard deviations
			# Set minimum values for the motion threshold and maximum value
			# in the subjects artifacts data for it to trigger motion correction
			# Note: Searching for data to be motion corrected must happen before initial drop
			# of data from the pandas dataframe due to indexing issues.
			if self.motion_threshold >= 0.0:	
				self.unneeded = self.unneeded + [ind for ind, accel in enumerate(self.eda_signals['EDA_Artifacts']) if accel >= self.motion_threshold and ind not in self.unneeded]


		# Process PPG data using neurokit2 ppg processing library
		ppg_raw = [float(signal[20]) for signal in self.data] # Convert PPG string to floats
		self.ppg_signals, self.ppg_info = nk.ppg_process(ppg_raw, sampling_rate=self.sr) # Process ppg data
		self.ppg_analysis = nk.ppg_analyze(self.ppg_signals, sampling_rate=self.sr) # Analyze the cleaned ppg data for HRV metrics

		# Drop all of the samples that aren't of interest or have motion artifacts in them
		self.eda_signals.drop(self.unneeded, inplace = True) # Drop EDA samples
		self.ppg_signals.drop(self.unneeded, inplace = True) # Drop PPG samples
		for ind in sorted(self.unneeded, reverse = True): # Remove labels that are not of interest or during a motion artifact in reverse order
			del self.labels[ind] # Drop label
		if len(self.labels) == 0: # If no events of interest in shimmer data
			self.failed.append(self.subject_ID + ' events of interest not found in shimmer data')
			return # skip processing for subject
		self.labels = np.array([datum for datum in self.labels]) # Reorganizing data in a numpy array

		# Call to breakdown function to create summaries of eda and ppg data by page and text
		self.eda_summary = self.breakdown(self.eda_signals['EDA_Clean'], 'EDA', self.data_timestamps, artifacts = self.eda_signals['EDA_Artifacts'], y_label = 'Skin Conductance (uS)')
		self.ppg_summary = self.breakdown(self.ppg_signals['PPG_Clean'], 'PPG', self.data_timestamps, y_label = 'PPG (au)')

		self.artifact_summary = self.summarize(self.eda_signals['EDA_Artifacts']) # Update summary statistics of EDA artifacts to check for abnormal motion in data of interest
		if self.artifact_summary[3] >= self.max_motion_artifact: # Check if max motion artifact is above maximum acceptable levels
				self.warning.append(self.subject_ID + ' shimmer has motion artifacts greater than 1') # Append to the warnings that this subject has significant motion in data


		self.save()# Save the data
		self.success.append(self.subject_ID) # Append subject ID onto successfully processed list

	def breakdown(self, data, data_type, timestamps, artifacts = [], plot = False, y_label = ''):
		#-------------------------- Breakdown Overview -------------------------#
		# This function will date a given dataset and loop through the experiment
		# texts and pages to give summaries at various levels (i.e. subject level,
		#  text level, page level).
		# If plot is passes as True in the function call, it will also save .pngs
		# of the page level data. If artifacts it will also include the artifact
		# in the plot at the expense of not being able to include a y-axis label
		# due to matplotlib being weird.
		#-----------------------------------------------------------------------#
		if path.exists(self.output_path + self.plot_path) == False: # Check to see if the folder already exists
			os.makedirs(self.output_path + self.plot_path) # If it doesn't exist, create the folder
		#------------ Summary Variable Format ------------#
		#self.summary['Overall'][0:3] = Entire dataset summary statistics
		#self.summary['Text Name'] - Dictionary call too grab text summaries i.e. 'Bias' or 'Variables'
		#self.summary['Text Name'][0:3] = text level summary - [mean of text data, standard deviation of text data, sum of text data, length of text data]
		#self.summary['Text Name'][4][Page Number - 1][0:3] = page level summarys - [0][mean of page data, standard deviation of page data, sum of page data, length of page data]
		self.summary = {text: [0, 0, 0, 0, [[0, 0, 0, 0] for page in self.pages]] for text in self.texts}# Declare hold summary variable to hold text and page level data
		self.summary['Overall'] = self.summarize(data)
		if len(artifacts) > 0:
			self.summary['Artifacts'] = self.summarize(artifacts)
		for text in self.texts: # Run through texts
			self.text_indices = self.search(self.labels[:, 0].tolist(), [text]) # Search for relavant text data
			self.text_data = np.take(data, self.text_indices) # Grab text data
			if len(artifacts) > 0: # If artifacts were passed into the call
				self.text_artifacts = np.take(artifacts, self.text_indices) # Take the text artificats of interest
			self.text_labels = np.take(self.labels, self.text_indices, axis = 0) # Grab text labels
			self.text_timestamps = np.take(timestamps, self.text_indices, axis = 0)
			self.text_summary = self.summarize(self.text_data) # Call to summarize function to grab relavant summary data
			self.summary[text][:4] = self.text_summary # Append text summary to the dictionary entry
			for page in self.pages: # Run through pages
				self.page_indices = self.search(self.text_labels[:, 1].tolist(), [str(page + 1)]) # Search for relavant text data
				self.page_data = np.take(self.text_data, self.page_indices)
				self.page_timestamps = np.take(self.text_timestamps, self.page_indices)
				self.page_start = self.page_timestamps[0]
				self.page_timestamps = [timestamp - self.page_start for timestamp in self.page_timestamps]
				self.summary[text][4][page - 1] = self.summarize(self.page_data)
				if len(artifacts) > 0:
					self.page_artifacts = np.take(self.text_artifacts, self.page_indices)
					fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True) # Declare the subplots
					self.page_data.plot(ax = axes[0], kind = 'line', title = self.subject_ID + ' - ' + text + ' - Page ' + str(page + 1), y = self.page_timestamps) # Plot the data
					self.page_artifacts.plot(ax = axes[1], kind = 'line', title = self.subject_ID + ' - ' + text + ' - Page ' + str(page + 1) + ' - Artifacts', y = self.page_timestamps) # Plot the artifacts
					axes[1].set_ylim(self.summary['Artifacts'][2:4]) # Set the artifact y limit to the min and max of the subjects artifact data
				else: # If artifacts were passed into the function call
					self.page_data.plot(kind ='line', title = self.subject_ID + ' - ' + text + ' - Page ' + str(page + 1), y = self.page_timestamps) # Plot the data
					plt.ylabel(y_label) # Declare the y label
				plt.xlabel('Sample') # Declare the x label
				plt.savefig(self.output_path + self.plot_path + self.subject_ID + '-' + data_type + '-' + text + '-' + str(page + 1) + '.png')
				plt.close() # Close the plot to conserve on memory
		return self.summary # Return the summary


	def detect_motion_artifacts(self, signal):
		self.signal = signal
		self.kernel = [-1,1]
		self.first_elt = self.signal[0]
		self.last_elt = self.signal[len(self.signal) - 1] # LINE TO CHECK - This might be messing the data up -- Had to change due to it throwing an error
		self.signal = np.array([self.first_elt, self.first_elt] + self.signal.tolist() + [self.last_elt, self.last_elt]) # pad ends for convolution to prevent edge artifacts
		self.signal_conv = abs(np.convolve(self.signal, self.kernel))
		self.signal_conv = self.signal_conv[2:-3] # remove padded elts - had to remove an extra index and removed the last since [2:x]
		return self.signal_conv


	def plot(self, data, range = None, title = None, x_label = None, y_label = None):
		if range == None: # If no range given, plot the entire data
			self.doi = data
			if title == None:
				title = self.subject_ID + ' - All Data'
			self.doi.plot(kind = 'line', title = title)
			self.doi_filename = self.subject_ID + '_DOI.png';
		else: # If a range is given
			self.ioi = [ind for ind in data.index if ind in range] # Find all indexes in the data that are within range
			self.doi = data.loc[self.ioi]
			if title == None:
				title = self.subject_ID + ' - Indices ' + str(range[0]) + '-' + str(range[len(range) - 1])
			self.doi.plot(kind = 'line', title = title)
			self.doi_filename = self.subject_ID + '_ROI_' + str(range[0] + 1) + '-' + str(range[len(range) - 1] + 1) + '.png'
		if x_label != None:
			plt.xlabel(x_label)
		if y_label != None:
			plt.y_label(y_label)
		plt.show()
		request = input('Figure generated, would you like to save the output? (Y/N)\n')
		if request == 'Y' or request == 'yes':
			plt.savefig(self.folder_prefix + self.subject_ID + self.folder_suffix + self.plot_path + self.doi_filename)
			print('Plot saved to ' + self.folder_prefix + self.subject_ID + self.folder_suffix + self.plot_path + self.doi_filename)
		plt.close()
		return

	def save(self):
		self.output_path = self.folder_prefix + self.subject_ID + self.folder_suffix
		self.eda_signals.to_csv(self.output_path + self.subject_ID + '_EDA_signals.csv', index_label = 'Sample Index') # Save EDA signals
		self.save_pickle(self.eda_info, self.output_path + self.subject_ID + '_EDA_info.pkl') # Save EDA info
		self.eda_data.to_csv(self.output_path + self.subject_ID + '_EDA_data.csv', index_label = 'Sample Index') # Save EDA data
		self.save_pickle(self.eda_summary, self.output_path + self.subject_ID + '_EDA_summary.pkl')# Save EDA summary
		self.ppg_signals.to_csv(self.output_path + self.subject_ID + '_PPG_signals.csv', index_label = 'Sample Index')# Save PPG Signals
		self.save_pickle(self.ppg_info, self.output_path + self.subject_ID + '_PPG_info.pkl')# Save PPG Info
		self.ppg_analysis.to_csv(self.output_path + self.subject_ID + '_PPG_analysis.csv', index_label = 'Sample Index')# Save PPG Analysis
		self.save_pickle(self.ppg_summary, self.output_path + self.subject_ID + '_PPG_summary.pkl')# Save PPG Summary
		np.save(self.output_path + self.subject_ID + '_shimmer_labels.npy', self.labels)# Save Physio Labels
		return

	def load(self, subject):
		self.subject_ID = 'EML1_' + '0'*(3 - len(str(subject))) + str(subject)
		self.output_path = self.folder_prefix + self.subject_ID + self.folder_suffix
		if path.exists(self.output_path + self.subject_ID + '_labels.npy') == True and reprocess == False: # If it exists and reprocessing not allowed
			print(self.subject_ID + ' physio data loading failed due to data not being processed')
			return # halt processing of this subject
		self.eda_signals = pd.read_csv(self.output_path + self.subject_ID + '_EDA_signals.csv', ',')
		self.eda_signals.set_index('Sample Index', inplace=True)
		self.eda_info = pd.read_pickle(self.output_path + self.subject_ID + '_EDA_info.pkl')
		self.eda_data = pd.read_csv(self.output_path + self.subject_ID + '_EDA_data.csv', ',')
		self.eda_data.set_index('Sample Index', inplace=True)
		self.eda_summary = self.read_pickle(self.output_path + self.subject_ID + '_EDA_summary.pkl')
		self.ppg_signals = pd.read_csv(self.output_path + self.subject_ID + '_PPG_signals.csv', ',')
		self.ppg_signals.set_index('Sample Index', inplace=True)
		self.ppg_info = pd.read_pickle(self.output_path + self.subject_ID + '_PPG_info.pkl')
		self.ppg_analysis = pd.read_csv(self.output_path + self.subject_ID + '_PPG_analysis.csv', ',')
		self.ppg_analysis.set_index('Sample Index', inplace=True)
		self.ppg_summary = self.read_pickle(self.output_path + self.subject_ID + '_PPG_summary.pkl')
		self.labels = np.load(self.output_path + self.subject_ID + '_shimmer_labels.npy')

	def summarize(self, data):
		return [data.mean(), data.std(), data.min(), data.max()]

	def search(self, data, identity, inverse = False):
		if inverse == False: # If your looking for anything matching an item in the idenity list
			return [ind for ind, datum in enumerate(data) if datum in identity]
		else: # If your looking for anything other than items matching something in the idenity list
			return [ind for ind, datum in enumerate(data) if datum not in identity]

	def read_csv(self, folder, filename, delim):
		try:
			with open(folder + filename) as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=delim)
				return np.array([line for line in csv_reader])
		except:
			return []

	def save_csv(self, data, path, keys):
		print(data)
		with open(path, 'wb') as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames = keys)
			writer.writeheader()
			writer.writerows(data)

	def save_pickle(self, data, path):
		with open(path, 'wb') as cucumber:
			pickle.dump(data, cucumber, protocol=pickle.HIGHEST_PROTOCOL)

	def read_pickle(self, path):
		with open(path, 'rb') as cucumber:
			data = pickle.load(cucumber)
		return


# ---------- Shimmer Raw File Contents - self.data -------- #
# 0 - Timestamp
# 1 - Accel_LN_X_CAL
# 2 - Accel_LN_Y_CAL
# 3 - Accel_LN_Z_CAL
# 4 - Accel_WR_X_CAL
# 5 - Accel_LN_Y_CAL
# 6 - Accel_LN_Z_CAL
# 7 - Battery
# 8 - Ext_Exp_A15_CAL
# 9 - Ext_Exp_A6_CAL
# 10 - Ext_Exp_A7_CAL
# 11 - GSR_Range_CAL
# 12 - GSR_Skin_Conductance_CAL
# 13 - GSR_Skin_Resistance_CAL
# 14 - Gyro_X_CAL
# 15 - Gyro_Y_CAL
# 16 - Gyro_Z_CAL
# 17 - Mag_X_CAL
# 18 - Mag_Y_CAL
# 19 - Mag_Z_CAL
# 20 - PPG_A13_CAL
# 21 - Pressure_BMP280_CAL
# 22 - Temperature_BMP280_CAL

# ------------- EDA Info Contents - self.eda_info -------------- #
# 0 - SCR Onsets
# 1 - SCR Peaks
# 2 - SCR Height
# 3 - SCR Amplitude
# 4 - SCR Rise Time
# 5 - SCR Recovery Time

# ------------- PPG Info Contents - self.ppg_info-------------- #
# 0 - PPG Peaks
# 1 - Sampling Rate

# ----------- PPG Analysis Contents - self.ppg_analysis------------ #
# 0 - PPG_Rate_Mean
# 1 - HRV_RMSSD
# 2 - HRV_MeanNN
# 3 - HRV_SDNN
# 4 - HRV_SDSD
# 5 - HRV_CVNN
# 6 - HRV_CVSD
# 7 - HRV_MedianNN
# 8 - HRV_MadNN
# 9 - HRV_MCVNN
# 10 - HRV_IQRNN
# 11 - HRV_pNN50
# 12 - HRV_pNN20
# 13 - HRV_TINN
# 14 - HRV_HTI
# 15 - HRV_ULF
# 16 - HRV_VLF
# 17 - HRV_LF
# 18 - HRV_HF
# 19 - HRV_VHF
# 20 - HRV_LFHF
# 21 - HRV_LFn
# 22 - HRV_HFn
# 23 - HRV_LnHF
# 24 - HRV_SD1
# 25 - HRV_SD2
# 26 - HRV_SD1SD2
# 27 - HRV_S
# 28 - HRV_CSI
# 29 - HRV_CVI
# 30 - HRV_CSI_Modified
# 31 - HRV_PIP
# 32 - HRV_IALS
# 33 - HRV_PSS
# 34 - HRV_PAS
# 35 - HRV_GI
# 36 - HRV_SI
# 37 - HRV_AI
# 38 - HRV_PI
# 39 - HRV_C1d
# 40 - HRV_C1a
# 41 - HRV_SD1d
# 42 - HRV_SD1a
# 43 - HRV_C2d
# 44 - HRV_C2a
# 45 - HRV_SD2d
# 46 - HRV_SD2a
# 47 - HRV_Cd
# 48 - HRV_Ca
# 49 - HRV_SDNNd
# 50 - HRV_SDNNa
# 51 - HRV_ApEn
# 52 - HRV_SampEn
# 53 - HRV_MSE
# 54 - HRV_CMSE
# 55 - HRV_RCMSE
# 56 - HRV_DFA
# 57 - HRV_CorrDim
