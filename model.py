import numpy as np
import os
import activations


class NeuralNetwork:
    
    def __init__(self, sizes):
        '''
        初始化神经网络
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        # 初始化权重，输入层没有权重和偏置
        self.weights = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.array([0])] + [np.random.randn(x, 1) for x in sizes[1:]]
        # 存储线性变换的结果
        self.linear_transforms = [np.zeros(bias.shape) for bias in self.biases]
        # 存储非线性变换的结果
        self.activations = [np.zeros(bias.shape) for bias in self.biases]
    
    def forward(self, input):
        '''
        前向传播
        '''
        self.activations[0] = input
        for i in range(1, self.num_layers):
            # 线性变换
            self.linear_transforms[i] = self.weights[i].dot(self.activations[i-1]) + self.biases[i]
            # 非线性变换
            # 在最后一层使用softmax激活函数
            if i == self.num_layers-1:
                self.activations[i] = activations.softmax(self.linear_transforms[i])
            else:
                self.activations[i] = activations.relu(self.linear_transforms[i])
        return self.activations[-1]
    
    def backward(self, loss_gradient):
        '''
        反向传播   

        `loss_gradient`为损失函数的求导结果
        '''
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        nabla_b[-1] = loss_gradient
        nabla_w[-1] = loss_gradient.dot(self.activations[-2].transpose())

        for layer in range(self.num_layers-2, 0, -1):
            loss_gradient = np.multiply(
                self.weights[layer+1].transpose().dot(loss_gradient),
                activations.relu_gradient(self.linear_transforms[layer])
            )
            nabla_b[layer] = loss_gradient
            nabla_w[layer] = loss_gradient.dot(self.activations[layer-1].transpose())
        
        return nabla_b, nabla_w
    
    def save(self, filename):
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            linear_transforms=self.linear_transforms,
            activations=self.activations
        )
    
    def load(self, filename):
        npz_members = np.load(os.path.join(os.curdir, 'models', filename), allow_pickle=True)

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        self.linear_transforms = list(npz_members['linear_transforms'])
        self.activations = list(npz_members['activations'])