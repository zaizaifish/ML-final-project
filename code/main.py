import os
import sys
from PIL import Image 
import numpy as np
from dataloader import *
from svm import *
from resnet import *
from cnn import *

if __name__ == "__main__":
    model = sys.argv[-1]
    ### data loading
    # 1 for messy, 0 for clean
    train, val, test = parser()
    # parameters
    batch_size = 10
    repeat = 1
    shuffle = True
    train_loader, val_loader = load(train, val, batch_size, repeat, shuffle)
    
    result = None
    if model == "svm":
        # set hyper parameters
        parameters = dict(
            C = [1e-1, 1e-2, 1e-3],
            kernel = ['linear', 'poly', 'rbf'],
            degree = [3, 5]
        )
        result = SVM(train, val, test, parameters)
    elif model == "cnn":
        ### set hyper parameters
        parameters = dict(
            learning_rate = [1e-2, 1e-3, 1e-4, 1e-5],
            num_epochs = [10, 20, 30],
            criterion = [torch.nn.CrossEntropyLoss(), torch.nn.NLLLoss()]
        )
        result = CNN(train_loader, val_loader, test, parameters)
    else:
        ### set hyper parameters
        parameters = dict(
            learning_rate = [1e-2, 1e-3, 1e-4, 1e-5],
            num_epochs = [10, 20, 30],
            criterion = [torch.nn.CrossEntropyLoss(), torch.nn.NLLLoss()]
        )
        result = RESNET(train_loader, val_loader, test, parameters)
    print('Test Result: {}'.format(result))
    true = np.array([0,0,1,0,1,1,0,1,1,0])
    accuracy = 1 - np.sum(np.abs(true - result)) / result.shape[0]
    print('Accuracy Result: {}'.format(round(accuracy,2)))
