### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, load_testing_images, train_valid_split
from Configure import model_configs, training_configs
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir",default='../data', help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		print("--- Training ---")
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
		print("Shape of X_train:", x_train.shape)
		print("Shape of y_train:", y_train.shape)
		print("Shape of X_val:", x_valid.shape)
		print("Shape of y_val:", y_valid.shape)
		print("Shape of X_test:", x_test.shape)
		print("Shape of y_test:", y_test.shape)

		# Training and validation
		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_valid, y_valid, [110, 120, 130, 140, 150])

	elif args.mode == 'test':
		print("--- Testing ---")
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test, [110, 120, 130, 140, 150])

	elif args.mode == 'predict':
		print("--- Predicting on Private Test set ---")
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)
		# visualizing the testing image to check image shape
		visualize(x_test[0], 'Test-Image-1.png')
		visualize(x_test[1], 'Test-Image-2.png')
		# Predicting and storing results on private testing dataset
		predictions = model.predict_prob(x_test)
		np.save(model_configs['result_dir'], predictions)


### END CODE HERE
