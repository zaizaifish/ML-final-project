from sklearn import svm
import time
import numpy as np
import copy
from itertools import product
def SVM(train, val, test, parameters):
    # reshape input for svm
    x_train = list()
    y_train = list()
    for data in train:
        image = data[0]
        label = data[1]
        x_train.append(image)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0],-1)
    
    x_val = list()
    y_val = list()
    for data in train:
        image = data[0]
        label = data[1]
        x_val.append(image)
        y_val.append(label)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_val = x_val.reshape(x_val.shape[0],-1)
    
    test = test.reshape(test.shape[0],-1)
    
    
    parameters = [dict(zip(parameters, v)) for v in product(*parameters.values())]
    best_model, best_acc = None, 0
    for index, cur_dict in enumerate(parameters):
        print("Parameter Setting: ", index)
        print(cur_dict)

        kernel = cur_dict['kernel'] # radical basis function
        C = cur_dict['C'] # punishment rate
        degree = cur_dict['degree']
        decision_function_shape = 'ovo' # one-vs-one strategy
        model = svm.SVC(kernel = kernel, C = C, decision_function_shape = decision_function_shape, degree = degree)
        # train model
        start_time = time.time()
        model.fit(x_train, y_train)
        training_time = time.time()-start_time
        print("Training time: {}".format(round(training_time,2)))

        # val 
        y_val_pred = model.predict(x_val)
        result_val = 1 - np.sum(np.abs(y_val - y_val_pred)) / y_val.shape[0]
        if result_val > best_acc:
            best_acc = result_val
            best_model = copy.deepcopy(model)
    # test
    y_test_pred = best_model.predict(test)
    return y_test_pred