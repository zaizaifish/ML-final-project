### Dependencies:
sklearn == 0.22.2
numpy == 1.18.5
matplotlib == 3.0.3
torch == 1.11.0
tensorboard == 2.8.0
tensorboardX == 2.5
  
### Dataset
Input data are stroed in ./dataset. There are 3 datasets as train, val, test.

### To run
Run this in commnd line
`
python model.py model_name
`

If model_name is not specified, main function will automatically select resnet as model. 

To run svm:
`
python model.py model_name svm
`

To run cnn:
`
python model.py model_name cnn
`
