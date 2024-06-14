
### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
    "learning_rate":0.0005,
	"model_dir": '../saved_models',
    "test_model_dir": '../saved_models',
    "result_dir": '../results/predictions.npy',
	# ...
}

training_configs = {
	"learning_rate": 0.0005,
    "model_dir": '../saved_models',
    "test_model_dir": '../saved_models',
	"max_epochs": 150,
	"batch_size": 256,
	"save_interval":10
	# ...
}

### END CODE HERE
