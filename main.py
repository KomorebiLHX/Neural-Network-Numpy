import numpy as np
import pandas as pd
import utils
from model import NeuralNetwork
from optimizer import SGD

np.random.seed(42)


batch_size = 16
epochs= 80
layers = [[784, 20, 10], [784, 30, 10], [784, 40, 10]]
learning_rates = [5e-3, 1e-2, 2e-2]
weight_decaies = [0, 1e-2, 2e-2] # L2 Penalty

print('Loading dataset...')
train_data, val_data, test_data = utils.load_mnist()

print('Training models...')
# 参数查找
best_config = {'accuracy': 0}
for layer in layers:
    for learning_rate in learning_rates:
        for weight_decay in weight_decaies:
            print(f"**Current layer: {layer}, Current learning rate: {learning_rate}, Current weight decay: {weight_decay}")
            model = NeuralNetwork(layer)
            optimizer = SGD(model, learning_rate, weight_decay, batch_size)
            accuracy = utils.fit(model, optimizer, train_data, val_data, epochs)
            if accuracy > best_config['accuracy']:
                best_config['accuracy'] = accuracy
                best_config['layer'] = layer
                best_config['learning_rate'] = learning_rate
                best_config['weight_decay'] = weight_decay


# best_config = {'accuracy': 96.77,
# 'layer': [784, 40, 10],
# 'learning_rate': 0.02,
# 'weight_decay': 0}

print("Testing...")
model = NeuralNetwork(best_config['layer'])
model.load(f"model_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.npz")
utils.test(model, test_data)

print("Visualizing...")
log = pd.read_csv(f"logs/log_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.csv")
utils.visualize(model, log)
