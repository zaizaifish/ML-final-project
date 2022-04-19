import copy
import numpy as np 
import torch
import torch.nn as nn
import time
import torchvision.models as models
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from itertools import product

def RESNET(train_loader, val_loader, test, parameters):
    ### define neural network structure
    ### use pretrained Resnet
    class RESNET(nn.Module):
        def __init__(self):
            super(RESNET, self).__init__()
            self.norm = nn.BatchNorm2d(num_features=3,affine=True)
            self.resnet = models.resnet18(pretrained=True)

            self.fc1 = nn.Linear(in_features=1000, out_features=32)
            self.fc2 = nn.Linear(in_features=32, out_features=16)
            self.fc3 = nn.Linear(in_features=16, out_features=2)

        def forward(self, x):
            x = self.norm(x)
            x = self.resnet(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x), dim = 1)
            return x
    ### unpack hyper parameters
    parameters = [dict(zip(parameters, v)) for v in product(*parameters.values())]
    best_model, best_acc, best_loss_record = None, 0, []
    best_parameter = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index, cur_dict in enumerate(parameters):
        print("Parameter Setting: ", index)
        print(cur_dict)
        num_epochs = cur_dict["num_epochs"]
        learning_rate = cur_dict["learning_rate"]
        criterion = cur_dict["criterion"]
        ### define model, loss function and optimizer
        model = RESNET().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        ### train step
        start_time = time.time()
        train_model, train_loss, train_loss_record = None, float("inf"), []
        loss_first = 0
        for t in range(num_epochs):
            loss_record = list()
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)        
                labels = labels.squeeze()                    
                loss = criterion(outputs, labels.long())
                loss_record.append(loss.item()) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            ### save best
            loss_cnt = sum(loss_record)
            if loss_cnt < train_loss:
                train_loss = loss_cnt
                train_model = copy.deepcopy(model)
            if (t == 0): loss_first = loss_cnt
            print("Epoch ", t, "Loss: ", loss_cnt)
            train_loss_record.append(loss_cnt)
        model = train_model
        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))
        ### val step
        cnt = 0
        error = 0

        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)        
            labels = labels.squeeze()      
            cnt += labels.shape[0]        
            error += torch.sum(torch.abs(torch.argmax(outputs, dim = 1) - labels))
        accuracy = 1 - error / cnt
        print("val acc: ",accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = copy.deepcopy(model)
            best_loss_record = train_loss_record
            best_parameter = cur_dict
        ### free model memory after each train-val
        torch.cuda.empty_cache()
    ### print out best model and its info
    model = best_model
    print("Best Loss Record", best_loss_record)
    print("Best Parameter Setting", best_parameter)
    # make predictions
    test = torch.from_numpy(test).type(torch.Tensor)
    test = test.to(device)
    y_test_pred = model(test)
    print('Test Prob: {}'.format(y_test_pred))
    _, res = torch.max(y_test_pred,1)
    res = res.cpu().detach()
    res = res.numpy()

    # include tensorboard
    writer = SummaryWriter('../result') 
    loss = loss_first # loss for first time
    for i, (name, param) in enumerate(model.named_parameters()):
        writer.add_histogram(name, param, 0)
        writer.add_scalar('loss', loss, i)
        loss = loss * 0.5
    writer.add_graph(model, torch.rand([1,3,299,299]).to(device))
    writer.close()
    return res
