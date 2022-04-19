import numpy as np 
import torch
import torch.nn as nn
import time
import torchvision.models as models
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy
from itertools import product
def CNN(train_loader, val_loader, test, parameters):
    # input: 299*299*3
    # out_conv1 = (299 - 4) + 1 = 296
    # max_pool1 = 296 / 2 = 148
    # out_conv2 = (148 - 5) + 1 = 144
    # max_pool2 = 144 / 2 = 72
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.norm = nn.BatchNorm2d(num_features=3,affine=True)
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 4)
            self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
            self.fc1 = nn.Linear(in_features=16*72*72, out_features=32)
            self.fc2 = nn.Linear(in_features=32, out_features=16)
            self.fc3 = nn.Linear(in_features=16, out_features=2)

        def forward(self, x):
            x = self.norm(x)
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x), dim = 1)
            # x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]     
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = [dict(zip(parameters, v)) for v in product(*parameters.values())]
    best_model, best_acc, best_loss_record = None, 0, []
    best_parameter = {}
    for index, cur_dict in enumerate(parameters):
        print("Parameter Setting: ", index)
        print(cur_dict)
        num_epochs = cur_dict["num_epochs"]
        learning_rate = cur_dict["learning_rate"]
        criterion = cur_dict["criterion"]
        ### define model, loss function and optimizer
            
        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
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