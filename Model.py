### YOUR CODE HERE
import torch
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record, preprocess_for_testing
import torch
import torch.nn as nn
import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.network = self.network.cuda()
        lr = self.configs['learning_rate']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)


    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        print("--- Training ---")
        self.network.train()

        print('train_configs',configs)
        batch_size = configs['batch_size']
        max_epochs = configs['max_epochs']
        save_interval = configs['save_interval']

        samples, _ = x_train.shape
        batches = samples // batch_size
        loss = 0

        for epoch in range(1, max_epochs + 1):
            time1 = time.time()

            shuffled_idx = np.random.permutation(samples)
            shuffled_x_train = x_train[shuffled_idx]
            shuffled_y_train = y_train[shuffled_idx]

            for b in range(batches):
                shuffled_x_batch = [parse_record(x, True) for x in shuffled_x_train[b * batch_size: (b+1) * batch_size]]
                shuffled_y_batch = shuffled_y_train[b * batch_size: (b+1) * batch_size]
                shuffled_x_batch_t = torch.stack((shuffled_x_batch)).float().cuda()
                shuffled_y_batch_t = torch.tensor((shuffled_y_batch)).float().cuda()

                model = self.network.cuda()
                y_preds = model(shuffled_x_batch_t)

                self.optimizer.zero_grad()
                loss = self.criterion(y_preds, shuffled_y_batch_t.long())
                loss.backward()
                self.optimizer.step()

                print(f"Batch: {b}/{batches} --- Loss: {loss}", end='\r', flush=True)

            time2 = time.time()
            duration = time2 - time1
            print(f"Epoch: {epoch} --- Loss: {loss} --- Duration: {duration}")

            self.scheduler.step()

            if epoch % save_interval == 0:
                self.save_model(epoch)

    def evaluate(self, x, y,checkpoints):
        print("--- Evaluation ---")
        self.network.eval()
        for checkpoint in checkpoints:
            model_dir = self.configs['model_dir']
            checkpoint_model = os.path.join(f"{model_dir}", 'my-model-%d.ckpt' % checkpoint)
            self.load_model(checkpoint_model)

            predictions = list()
            samples, _ = x.shape
            for i in tqdm.tqdm(range(samples)):
                img = parse_record(x[i], False).float().to('cuda').view(1, 3, 32, 32)
                logits = self.network(img)
                pred = int(torch.max(logits.data, 1)[1])
                predictions.append(pred)

            print(f"Test accuracy: {torch.sum( torch.tensor(predictions) == torch.tensor(y) ) / torch.tensor(y).shape[0]}")


    def predict_prob(self, x):
        print("--- Evaluation for Private Dataset ---")
        self.network.eval()
        model_dir = self.configs['test_model_dir']
        checkpoint_model = os.path.join(f"{model_dir}", 'my-model-%d.ckpt' % 150)
        self.load_model(checkpoint_model)

        predictions = list()
        samples, _ = x.shape
        for i in tqdm.tqdm(range(samples)):
            img = x[i].reshape((32, 32, 3))
            img = preprocess_for_testing(img).float().to('cuda').view(1, 3, 32, 32)
            pred = self.network(img)
            predictions.append(pred.cpu().detach().numpy())

        predictions = np.array(predictions)
        x, y, z = predictions.shape
        print("x, y, z", x, y, z)
        predictions = predictions.reshape((x, y * z))
        exp_predictions = np.exp(predictions)
        summed_exp_predictions = exp_predictions.sum(axis=1)
        logits = (exp_predictions.T / summed_exp_predictions).T

        print("Verification: shape of predictions is", logits.shape)

        return np.array(logits)

    def save_model(self, epoch):
        model_dir = self.configs['model_dir']
        pwd = os.path.join(f"{model_dir}", 'my-model-%d.ckpt' % epoch)
        os.makedirs(f"{model_dir}", exist_ok=True)
        torch.save(self.network.state_dict(), pwd)
        print(f"--- Saving model at epoch {epoch} ---")

    def load_model(self, checkpoint_model_epoch):
        loaded_model = torch.load(checkpoint_model_epoch, map_location='cpu')
        self.network.load_state_dict(loaded_model)
        print(f"--- Loading model {checkpoint_model_epoch} ---")


### END CODE HERE
