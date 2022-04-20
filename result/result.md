### Validation Accuracy for Different Parameters

#### SVM

| C (L2-Regularization)      | kernel | degree(for poly) | acc | 
| :-----------: | :-----------: | :-----------: | :-----------: |
|  0.1   |   linear   |  | 1.0  |
|  0.1   |    poly    | 3| 0.61 |
|  0.1   |    poly    | 5| 0.6  |
|  0.1   |    rbf     |  | 0.79 |
|  0.01  |   linear   |  | 1.0  |
|  0.01  |    poly    | 3| 0.54 |
|  0.01  |    poly    | 5| 0.52 |
|  0.01  |    rbf     |  | 0.79 |
|  0.001 |   linear   |  | 1.0  |
|  0.001 |    poly    | 3| 0.54 |
|  0.001 |    poly    | 5| 0.52 |
|  0.001 |    rbf     |  | 0.79 |

#### CNN

| learning_rate      | num_epochs | criterion | acc | 
| :-----------: | :-----------: | :-----------: | :-----------: |
|  0.01  |  10  | CrossEntropyLoss() | 0.5  |
|  0.01  |  10  |     NLLLoss()      | 0.5  |
|  0.01  |  20  | CrossEntropyLoss() | 0.5  |
|  0.01  |  20  |     NLLLoss()      | 0.5  |
|  0.01  |  30  | CrossEntropyLoss() | 0.5  |
|  0.01  |  30  |     NLLLoss()      | 0.5  |
|  0.001 |  10  | CrossEntropyLoss() | 0.75 |
|  0.001 |  10  |     NLLLoss()      | 0.5  |
|  0.001 |  20  | CrossEntropyLoss() | 0.7  |
|  0.001 |  20  |     NLLLoss()      | 0.5  |
|  0.001 |  30  | CrossEntropyLoss() | 0.65 |
|  0.001 |  30  |     NLLLoss()      | 0.6  |
|  0.0001|  10  | CrossEntropyLoss() | 0.8  |
|  0.0001|  10  |     NLLLoss()      | 0.65 |
|  0.0001|  20  | CrossEntropyLoss() | 0.65 |
|  0.0001|  20  |     NLLLoss()      | 0.6  |
|  0.0001|  30  | CrossEntropyLoss() | 0.55 |
|  0.0001|  30  |     NLLLoss()      | 0.55 |
|  0.00001|  10  | CrossEntropyLoss() | 0.7  |
|  0.00001|  10  |     NLLLoss()      | 0.6  |
|  0.00001|  20  | CrossEntropyLoss() | 0.75 |
|  0.00001|  20  |     NLLLoss()      | 0.6  |
|  0.00001|  30  | CrossEntropyLoss() | 0.75 |
|  0.00001|  30  |     NLLLoss()      | 0.7  |

#### ResNet

| learning_rate      | num_epochs | criterion | acc | 
| :-----------: | :-----------: | :-----------: | :-----------: |
|  0.01   |  10  | CrossEntropyLoss() | 0.5  |
|  0.01   |  10  |     NLLLoss()      | 0.5  |
|  0.01   |  20  | CrossEntropyLoss() | 0.5  |
|  0.01   |  20  |     NLLLoss()      | 0.5  |
|  0.01   |  30  | CrossEntropyLoss() | 0.5  |
|  0.01   |  30  |     NLLLoss()      | 0.5  |
|  0.001  |  10  | CrossEntropyLoss() | 1.0  |
|  0.001  |  10  |     NLLLoss()      | 0.85 |
|  0.001  |  20  | CrossEntropyLoss() | 0.65 |
|  0.001  |  20  |     NLLLoss()      | 0.8  |
|  0.001  |  30  | CrossEntropyLoss() | 0.9  |
|  0.001  |  30  |     NLLLoss()      | 0.95 |
|  0.0001 |  10  | CrossEntropyLoss() | 0.95 |
|  0.0001 |  10  |     NLLLoss()      | 0.95 |
|  0.0001 |  20  | CrossEntropyLoss() | 1.0  |
|  0.0001 |  20  |     NLLLoss()      | 0.95 |
|  0.0001 |  30  | CrossEntropyLoss() | 1.0  |
|  0.0001 |  30  |     NLLLoss()      | 1.0  |
|  0.00001|  10  | CrossEntropyLoss() | 1.0  |
|  0.00001|  10  |     NLLLoss()      | 1.0  |
|  0.00001|  20  | CrossEntropyLoss() | 1.0  |
|  0.00001|  20  |     NLLLoss()      | 0.95 |
|  0.00001|  30  | CrossEntropyLoss() | 1.0  |
|  0.00001|  30  |     NLLLoss()      | 1.0  |

### Best Parameter with Test Accuracy

| model |    |  |  | acc | 
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| svm |  'C': 0.1  | 'kernel': 'linear' | 'degree': none | 0.6 | 
| cnn |  'learning_rate': 0.0001  | 'num_epochs': 10 | 'criterion': CrossEntropyLoss() | 0.7 | 
| resnet |  'learning_rate': 0.001  | 'num_epochs': 10 | 'criterion': CrossEntropyLoss() | 1.0 | 
