from __future__ import absolute_import, division, print_function, unicode_literals
import os, shutil, nilearn, time, math, reader, time, csv, random, sklearn, config
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Ask tensorflow to not use GPU
import SimpleITK as sitk
import numpy as np
from numpy import asarray
import tensorflow as tf
import nibabel as nib
import plotly.graph_objects as go
from nilearn import image, plotting, datasets, surface
from nilearn.input_data import NiftiMasker
from keras.utils import to_categorical
from keras import models
from keras.layers import Layer
from keras.optimizers import SGD
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv3D, LeakyReLU
from random import randint, randrange

class KleinNet:

	def __init__(self):
		print("\n - KleinNet Initialized -\n - Process PID - " + str(os.getpid()) + ' -\n')
		os.chdir('../..')
		print('Current working directory change to ' + os.getcwd())
		try:
			os.chdir(config.result_directory + config.run_directory + "/Layer_1/")
			os.chdir("../../..")
		except:
			self.create_dir()

	def run(self):
		self.wrangle()
		self.build()
		self.train()
		self.test()
		self.plot_accuracy()
		for output in config.outputs:
			self.observe(output)

	def optimize(self):
		alpha_opt = [0.1, 0.01, 0.001, 0.0001]
		learning_rate_opt = [0.001, 1e-4, 1e-5, 1e-6]
		bias_opt = [0, 0.5, 1, 2]
		momentum_opt = [0, 0.001, 0.01, 0.1]
		epsilon_opt = [1e-5, 1e-6, 1e-7, 1e-8]
		index = 1
		for config.alpha in alpha_opt:
			for config.learning_rate in learning_rate_opt:
				for config.bias in bias_opt:
					for config.optimizer in config.optimizers:
						for config.epsilon, config.momentum in epsilon_opt, momentum_opt:
							self.build()
							self.train()
							self.test()
							self.plot_accuracy(index)
							index += 1

	def jack_knife(self, Range = None):
		if Range == None:
			Range = range(1, config.subject_count)
		for self.jackknife in Range:
			print("Running Jack-Knife on Subject " + str(self.jackknife))
			self.wrangle(range(config.subject_count), self.jackknife)
			self.build()
			self.train()
			self.plot_accuracy()
			self.ROC()

	def ROC(self):
		self.probabilities = self.model.predict(self.x_test).ravel()
		np.save(config.result_directory + config.run_directory + '/Jack_Knife/Probabilities/Sub-' + str(self.jackknife) + '_Volumes_Prob.np', self.probabilities)
		fpr, tpr, threshold = roc_curve(self.y_test, self.probabilities)
		predictions = np.argmax(self.probabilities, axis=-1)
		AUC = auc(fpr, tpr)
		plt.figure()
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr, tpr, label = 'RF (area = {:.3f})'.format(AUC))
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Subject ' + str(self.jackknife) + ' ROC Curve')
		plt.legend(loc = 'best')
		plt.savefig(config.result_directory + config.run_directory + "/Jack_Knife/Sub_" + str(self.jackknife) + "_ROC_Curve.png")
		plt.close()


	def SVM(self):
		config.loss = 'hinge'
		config.output_activation = 'rbf'


	def orient(self):
		print("\nOrienting and generating KleinNet lexicons")
		self.subject_IDs = ["sub-" + '0'*(config.ID_len - len(str(ind))) + str(ind) for ind in range(1, config.subject_count + 1)]
		self.numpy_folders = [config.data_directory  + subject_ID + '/' + config.numpy_output_dir + '/' for subject_ID in self.subject_IDs]
		self.volumes_filenames = [subject + "_volumes.npy" for subject in self.subject_IDs]
		self.labels_filenames = [subject + "_labels.npy" for subject in self.subject_IDs]
		self.header_filenames = [subject + "_headers.npy" for subject in self.subject_IDs]
		self.affines_filenames = [subject + "_affines.npy" for subject in self.subject_IDs]
		self.anat_folders = [config.data_directory + subject_ID + '/anat/' for subject_ID in self.subject_IDs]
		self.anat_filenames = [subject_ID + "_T1w.nii" for subject_ID in self.subject_IDs]

	def wrangle(self, subject_range = range(config.subject_count), jackknife = None):
		try:
			self.numpy_folders
		except:
			self.orient()
		self.progress_bar(0, (config.subject_count - 1), prefix = 'Wrangling Data', suffix = 'Complete', length = 40)
		for subject_index in subject_range:
			if subject_index != jackknife:
				image = np.load(self.numpy_folders[subject_index] + self.volumes_filenames[subject_index]) # Load data
				if config.wumbo == False:
					label = np.load(self.numpy_folders[subject_index] + self.labels_filenames[subject_index])
				else:
					label = np.random.randint(2, size = image.shape[0])
				try:
					images = np.append(images, image, axis = 0)
					labels = np.append(labels, label)
				except:
					images = image
					labels = label
			self.progress_bar(subject_index, (config.subject_count - 1), prefix = 'Wrangling Data', suffix = 'Complete', length = 40)
		print("\n")
		if config.shuffle == True:
			images, labels = self.shuffle(images, labels)
		if jackknife == None:
			self.x_train = images[:(round((images.shape[0]/3)*2)),:,:]
			self.y_train = labels[:(round((len(labels)/3)*2))]
			self.x_test = images[(round((images.shape[0]/3)*2)):,:]
			self.y_test = labels[(round((len(labels)/3)*2)):]
		else:
			self.x_train = images
			self.y_train = labels
			self.x_test = np.load(self.numpy_folders[jackknife - 1] + self.volumes_filenames[jackknife - 1])
			self.y_test = np.load(self.numpy_folders[jackknife - 1] + self.labels_filenames[jackknife - 1])


	def wrangle_subject(subject):
		self.images = np.load(self.numpy_folders[subject - 1] + self.volumes_filenames[subject - 1])
		self.labels = np.load(self.numpy_folders[subject - 1] + self.labels_filenames[subject - 1])
		self.header = np.load(self.numpy_folders[subject - 1] + self.header_filenames[subject - 1])
		self.anatomy = np.load(self.anat_folders[subject - 1] + self.anat_filenames[subject - 1])

	def shuffle(self, images, labels):
		indices = np.arange(images.shape[0])
		np.random.shuffle(indices)
		images = images[indices, :, :, :]
		labels = labels[indices]
		return images, labels

	def plan(self):
		print("\nPlanning KleinNet model structure")
		self.filter_counts = []
		convolution_size = config.init_filter_count
		for depth in range(config.convolution_depth*2):
			self.filter_counts.append(convolution_size)
			convolution_size = convolution_size*2

		self.layer_shapes = []
		self.output_layers = []
		conv_shape = [config.x_size, config.y_size, config.z_size]
		conv_layer = 1
		for depth in range(config.convolution_depth):
			conv_shape = self.calcConv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 3
			conv_shape = self.calcConv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 4
			if depth < config.convolution_depth - 1:
				conv_shape = self.calcMaxPool(conv_shape)

		self.new_shapes = []
		for layer_ind, conv_shape in enumerate(self.layer_shapes):
			new_shape = self.calcConvTrans(conv_shape)
			for layer in range(layer_ind,  0, -1):
				new_shape = self.calcConvTrans(new_shape)
				if layer % 2 == 1 & layer != 1:
					new_shape = self.calcUpSample(new_shape)
			self.new_shapes.append(new_shape)

		for layer, plan in enumerate(zip(self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes)):
			print("Layer ", layer + 1, " (", plan[0], ")| Filter count:", plan[1], "| Layer Shape: ", plan[2], "| Deconvolution Output: ", plan[3])

	def calcConv(self, shape):
		return [(input_length - filter_length + (2*pad))//stride + 1 for input_length, filter_length, stride, pad in zip(shape, config.kernel_size, config.kernel_stride, config.padding)]

	def calcMaxPool(self, shape):
		return [(input_length - pool_length + (2*pad))//stride + 1 for input_length, pool_length, stride, pad in zip(shape, config.pool_size, config.pool_stride, config.padding)]

	def calcConvTrans(self, shape):
		if config.zero_padding == 'valid':
			return [round((input_length - 1)*stride + filter_length) for input_length, filter_length, stride in zip(shape, config.kernel_size, config.kernel_stride)]
		else:
			return [round(input_length*stride) for input_length, filter_length, stride in zip(shape, config.kernel_size, config.kernel_stride)]

	def calcUpSample(self, shape):
		return [round((input_length - 1)*(filter_length/stride)*2) for input_length, filter_length, stride in zip(shape, config.pool_size, config.pool_stride)]

	def optimum_bias(self):
		if correct < incorrect:
			return math.log((config.correct/config.incorrect), (2.78))
		else:
			return math.log((config.incorrect/config.correct), (2.78))

	def build(self):
		try:
			self.filter_counts
		except:
			self.plan()
		print('\nConstructing KleinNet model')
		self.model = tf.keras.models.Sequential() # Create first convolutional layer
		for layer in range(1, config.convolution_depth + 1): # Build the layer on convolutions based on config convolution depth indicated
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer*2 - 2], config.kernel_size, strides = config.kernel_stride, padding = config.zero_padding, input_shape = (config.x_size, config.y_size, 1), use_bias = True, kernel_initializer = config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(config.bias)))
			self.model.add(LeakyReLU(alpha = config.alpha))
			self.model.add(tf.keras.layers.BatchNormalization())
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer*2 - 1], config.kernel_size, strides = config.kernel_stride, padding = config.zero_padding, use_bias = True, kernel_initializer = config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(config.bias)))
			self.model.add(LeakyReLU(alpha = config.alpha))
			self.model.add(tf.keras.layers.BatchNormalization())
			if layer < config.convolution_depth:
				self.model.add(tf.keras.layers.MaxPooling3D(pool_size = config.pool_size, strides = config.pool_stride, padding = config.zero_padding, data_format = "channels_last"))
		if config.density_dropout[0] == True: # Add dropout between convolution and density layer
			self.model.add(tf.keras.layers.Dropout(config.dropout))
		self.model.add(tf.keras.layers.Flatten()) # Create heavy top density layers
		for density, dense_dropout in zip(config.top_density, config.density_dropout[1:]):
			self.model.add(tf.keras.layers.Dense(density, use_bias = True, kernel_initializer = config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(config.bias))) # Density layer based on population size of V1 based on Full-density multi-scale account of structure and dynamics of macaque visual cortex by Albada et al.
			self.model.add(LeakyReLU(alpha = config.alpha))
			if dense_dropout == True:
				self.model.add(tf.keras.layers.Dropout(config.dropout))
		if config.output_activation != 'rbf':
			self.model.add(tf.keras.layers.Dense(1, activation=config.output_activation)) #Create output layer
		else:
			self.model.add(RBFLayer(1, 0.5))
		self.model.build()
		self.model.summary()

		if config.optimizer == 'Adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate, epsilon = config.epsilon, amsgrad = config.use_amsgrad)
		if config.optimizer == 'SGD':
			optimizer = tf.keras.optimizers.SGD(learning_rate = config.learning_rate, momentum = config.momentum, nesterov = config.use_nestrov)
		self.model.compile(optimizer = optimizer, loss = config.loss, metrics = ['accuracy']) # Compile model and run
		print('\nKleinNet model compiled using', config.optimizer)

	def train(self):
		self.history = self.model.fit(self.x_train, self.y_train, epochs = config.epochs, batch_size = config.batch_size, validation_data = (self.x_test, self.y_test))

	def test(self):
		self.loss, self.accuracy = self.model.evaluate(self.x_test,  self.y_test, verbose=2)

	def save(self):
		tf.save_model.save(self.model, 'Model_Description') # Save model

	def plot_accuracy(self, i = 1):
		print("\nEvaluating KleinNet model accuracy & loss...")
		for history_type in ['Accuracy', 'Loss']:		# Evaluate the model accuracy and loss
			plt.plot(self.history.history[history_type.lower()], label=history_type)
			plt.plot(self.history.history['val_' + history_type.lower()], label = 'Validation ' + history_type)
			plt.xlabel('Epoch')
			plt.ylabel(history_type)
			plt.legend(loc='upper right')
			plt.ylim([0, 1])
			title = "~learnig rate: " + str(config.learning_rate) + " ~alpha: " + str(config.alpha) + ' ~bias: ' + str(config.bias) + ' ~optimizer: ' + config.optimizer
			if config.optimizer == 'SGD':
				title = title + ' ~epsilon: ' + str(config.epsilon)
			else:
				title = title + ' ~momentum: ' + str(config.momentum)
			plt.title(title)
			plt.savefig(config.result_directory + config.run_directory + "/Model_Description/Model_" + str(i + 1) + "_" + history_type + ".png")
			plt.close()

	def observe(self, interest):
		print("\nObserving " + config.outputs_category[interest].lower() + " outcome structure")
		try:
			self.images
		except:
			self.wrangle(subject_range = [1])
		self.sample_label = -1
		while self.sample_label != interest: # Grab next sample that is the other category
			self.sample_label = self.labels[random.randint(self.images.shape[0])] # Grab sample label
		self.sample = self.images[rand_ind, :, :, :] # Grab sample volume
		#self.anatomie = self.anatomies[rand_ind]
		self.header = self.headers[rand_ind]
		self.category = config.outputs[sample_label]

		print("\nExtracting " + category + " answer features from KleinNet convolutional layers...")
		self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes
		layer_outputs = [layer.output for layer in self.model.layers[:]]

		for self.layer in range(1, (config.convolution_depth*2 + 1)): # Build deconvolutional models for each layer
			self.activation_model = tf.keras.models.Model(inputs = self.model.input, outputs = [layer_outputs[self.output_layers[self.layer - 1]], self.model.output])
			self.deconv_model = tf.keras.models.Sequential() # Create first convolutional layer
			self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, config.kernel_size, strides = config.kernel_stride, input_shape = (self.layer_shapes[self.layer - 1][0], self.layer_shapes[self.layer - 1][1], 1), kernel_initializer = tf.keras.initializers.Ones()))
			for deconv_layer in range(self.layer - 1, 0, -1): # Build the depths of the deconvolution model
				if deconv_layer % 2 == 1 & deconv_layer != 1:
					self.deconv_model.add(tf.keras.layers.UpSampling3D(size = config.pool_size, data_format = 'channels_last'))
				self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, config.kernel_size, strides = config.kernel_stride, kernel_initializer = tf.keras.initializers.Ones()))
			print('Summarizing layer ', self.layer, ' deconvolution model')
			self.deconv_model.build()
			self.deconv_model.summary()
			self.feature_maps, predictions = activation_model.predict(sample) # Grab feature map using single volume
			self.feature_maps = self.feature_maps[0, :, :, :].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2])

			self.progress_bar(0, self.feature_maps.shape[3] - 1, prefix = 'Extracting Layer ' + str(self.layer) + ' Features', suffix = 'Complete', length = 40)
			for self.map_index in range(self.feature_maps.shape[3]): # Save feature maps in glass brain visualization pictures
				feature_map = (self.feature_maps[:, :, map_index].reshape(self.current_shape[0], self.current_shape[1])) # Grab Feature map
				deconv_feature_map = self.deconv_model.predict(self.feature_maps[:, :, map_index].reshape(1, self.current_shape[0], self.current_shape[1], 1)).reshape(self.new_shape[0], self.new_shape[1])
				self.plot_all(heatmap, 'DeConv_Feature_Maps', map_index)
				self.progress_bar(map_index, self.feature_maps.shape[3] - 1, prefix = 'Extracting Layer ' + str(self.layer) + ' Features', suffix = 'Complete', length = 40)

			print("\n\nExtracting KleinNet model class activation maps for layer " + self.layer)
			with tf.GradientTape() as gtape: # Create CAM
				conv_output, predictions = activation_model(sample)
				loss = predictions[:, np.argmax(predictions[0])]
				grads = gtape.gradient(loss, conv_output)
				pooled_grads = K.mean(grads, axis = (0, 1, 2))

			self.heatmap = tf.math.reduce_mean((pooled_grads * conv_output), axis = -1)
			self.heatmap = np.maximum(self.heatmap, 0)
			max_heat = np.max(self.heatmap)
			if max_heat == 0:
				max_heat = 1e-10
			self.heatmap /= max_heat

			# Deconvolute heatmaps and visualize
			self.heatmap = deconv_model.predict(self.heatmap.reshape(1, self.current_shape[0], self.current_shape[1], 1)).reshape(self.new_shape[0], self.new_shape[1])
			self.plot_all(self.heatmap, 'CAM', 1)

	def plot_all(self, data, data_type, map_index):
		self.surf_stat_maps(data, data_type, map_index)
		#self.glass_brains(data, data_type, map_index)
		#self.stat_maps(data, data_type, map_index)

	def prepare_plots(self, data, data_type, map_index, plot_type):
		affine = self.header.get_best_affine()
		max_value, min_value, mean_value, std_value = describe_data(data)
		#-Thresholding could take some more consideration-#
		threshold = 0
		intensity = 0.5
		data = data * intensity
		# ---------------------------------------------- #
		data = nib.Nifti1Image(data, affine = self.affine, header = self.header) # Grab feature map
		title = layer + " " + data_type + " Map " + str(map_index) + " for  " + self.category + " Answer"
		output_folder = config.result_directory + config.run_directory + self.catergory + '/Layer_' + self.layer + '/' + data_type + '/' + plot_type + '/'
		return data, title, threshold, output_folder

	def glass_brains(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Glass_Brain")
		plotting.plot_glass_brain(stat_map_img = data, black_bg = True, plot_abs = False, display_mode = 'lzry', title = title, threshold = threshold, annotate = True, output_file = (output_folder + 'feature_' + str(map_index) + '-' + self.category + '_category.png')) # Plot feature map using nilearn glass brain - Original threshold = (mean_value + (std_value*2))


	def stat_maps(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Stat_Maps")
		for display, midfix, cut_coord in zip(['z', 'x', 'y'], ['-zview-', '-xview-', '-yview-'], [6, 6, 6]):
			plotting.plot_stat_map(data, bg_img = self.anatomy, display_mode = display, cut_coords = cut_coord, black_bg = True, title = title, threshold = threshold, annotate = True, output_file = (output_folder + 'feature_' + str(map_index) +  midfix + self.category + '_category.png')) # Plot feature map using nilearn glass brain

	def surf_stat_maps(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Surf_Stat_Maps")
		fsaverage = datasets.fetch_surf_fsaverage()

		texture = surface.vol_to_surf(data, fsaverage.pial_left)
		plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi = 'left', view = 'lateral', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_left, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-left-lateral-' + self.category + '_category.png'))
		plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi = 'left', view = 'medial', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_left, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-left-medial-' + self.category + '_category.png'))

		texture = surface.vol_to_surf(data, fsaverage.pial_right)
		plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi = 'right', view = 'lateral', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_right, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-right-lateral-' + self.category + '_category.png'))
		plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi = 'right', view = 'medial', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_right, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-right-medial-' + self.category + '_category.png'))


	def create_dir(self):
		first_dir = config.outputs_category # Create lists of all directory levels for extraction outputs
		second_dir = ['Layer_' + str(layer) for layer in range(1, config.convolution_depth*2 + 1)]
		third_dir = ["DeConv_Feature_Maps", "DeConv_CAM"]
		fourth_dir = ["GB", "SM", "SSM"]
		try:
			os.chdir(config.result_directory + config.run_directory + '/')
			os.chdir('../..')
			print('\nRun directory ' + config.result_directory + config.run_directory + ' currently exists, a clean run directory is needed for KleinNet to output results correctly, would you like to remove and replace the current run directory? (yes or no)')
			response = input()
			if response == 'yes':
				shutil.rmtree(config.result_directory + config.run_directory)
				time.sleep(1)
			else:
				return
		except:
			print('\nGenerating run directory')
		os.mkdir(config.result_directory + config.run_directory + '/')
		os.mkdir(config.result_directory + config.run_directory + "/Model_Description")
		os.mkdir(config.result_directory + config.run_directory + '/SVM')
		os.mkdir(config.result_directory + config.run_directory + '/Jack_Knife')
		os.mkdir(config.result_directory + config.run_directory + '/Jack_Knife/Probabilities')
		for first in first_dir:
			os.mkdir(config.result_directory + config.run_directory + "/" + first)
			for second in second_dir:
				os.mkdir(config.result_directory + config.run_directory + "/" + first + "/" + second)
				for third in third_dir:
					os.mkdir(config.result_directory + config.run_directory + "/" + first + "/" + second + "/" + third)
					for fourth in fourth_dir:
						os.mkdir(config.result_directory + config.run_directory + "/" + first + "/" + second + "/" + third + "/" + fourth)
		print('\nResult directories generated for ' + config.run_directory + '\n')


	def progress_bar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
		if iteration == total:
		    print()

	class RBFLayer(Layer):
	    def __init__(self, units, gamma, **kwargs):
	        super(RBFLayer, self).__init__(**kwargs)
	        self.units = units
	        self.gamma = K.cast_to_floatx(gamma)

	    def build(self, input_shape):
	        self.mu = self.add_weight(name='mu',
	                                  shape=(int(input_shape[1]), self.units),
	                                  initializer='uniform',
	                                  trainable=True)
	        super(RBFLayer, self).build(input_shape)

	    def call(self, inputs):
	        diff = K.expand_dims(inputs) - self.mu
	        l2 = K.sum(K.pow(diff,2), axis=1)
	        res = K.exp(-1 * self.gamma * l2)
	        return res

	    def compute_output_shape(self, input_shape):
	        return (input_shape[0], self.units)
