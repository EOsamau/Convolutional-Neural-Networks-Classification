### About Files

- **CNNModelBuild.ipynb** file has the initial model _experimentation and training phase_ to find the optimal CNN architecture and regularization for this classification
- **The CNNModelTrain.py** file is the script on the model build that _saves and stores the model weight and architecture_ as model.pt
- **model.pt** contains the trained weights and architecture of the model
- **MyTest.py** is a script that does not train the model but uses model.pt to evaluate a random test set for accuracy to ensure that the model actually works
