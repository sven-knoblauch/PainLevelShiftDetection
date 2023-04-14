import torch
from torch import nn


# #
#
# Torch model for embedding calculation. Hyperparameters to configure different architectures. 
# 
# #
class EmbeddingsModel(nn.Module):
    def __init__(self, model_hyperparameter):
        super(EmbeddingsModel, self).__init__()
        self.model_hyperparameter = model_hyperparameter
        self.sigmoid = self.model_hyperparameter["sigmoid"]
        self.dropout = self.model_hyperparameter["dropout"]
        self.layers = self.model_hyperparameter["layers"]

        #set layers according to hyperparameters
        self.embedding_model = torch.nn.Sequential()
        for idx in range(len(self.layers)-2):    
            self.embedding_model.add_module("linear_"+str(idx), nn.Linear(self.layers[idx], self.layers[idx+1]))
            self.embedding_model.add_module("batchnorm1d_"+str(idx), nn.BatchNorm1d(self.layers[idx+1]))
            self.embedding_model.add_module("relu_"+str(idx), nn.ReLU())
            self.embedding_model.add_module("dropout_"+str(idx), nn.Dropout(self.dropout))
            
        self.embedding_model.add_module("linear_output", nn.Linear(self.layers[-2], self.layers[-1]))

        #set activation function at the end
        if self.sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

    def forward(self, a):
        a = self.embedding_model(a)
        return self.activation(a)
  
# #
#
# Torch model for classification calculation. Hyperparameters to configure different architectures. 
# 
# #
class ClassificationModel(nn.Module):
    def __init__(self, model_hyperparameter):
        super(ClassificationModel, self).__init__()
        self.model_hyperparameter = model_hyperparameter
        self.dropout = self.model_hyperparameter["dropout"]
        self.layers = self.model_hyperparameter["layers"]
        self.head_type = self.model_hyperparameter["head_type"]

        #set layers according to hyperparameters
        self.classification_model = torch.nn.Sequential()
        for idx in range(len(self.layers)-1):    
            self.classification_model.add_module("linear_"+str(idx), nn.Linear(self.layers[idx], self.layers[idx+1]))
            self.classification_model.add_module("batchnorm1d_"+str(idx), nn.BatchNorm1d(self.layers[idx+1]))
            self.classification_model.add_module("relu_"+str(idx), nn.ReLU())
            self.classification_model.add_module("dropout_"+str(idx), nn.Dropout(self.dropout))
        
        #set classification head type for classification type or regression
        if self.head_type == 0:
            self.classification_model.add_module("linear_output", nn.Linear(self.layers[-1], 1))
            self.classification_model.add_module("sigmoid_activation", nn.Sigmoid())
        elif self.head_type == 1:
            self.classification_model.add_module("linear_output", nn.Linear(self.layers[-1], 1))
            self.classification_model.add_module("sigmoid_activation", nn.Tanh())
        else:
            self.classification_model.add_module("linear_output", nn.Linear(self.layers[-1], 3))
            self.classification_model.add_module("sigmoid_activation", nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.classification_model(x)



# #
#
# Torch model for siamese model. Using a embedding and classification model.
# Three different ways to calculate the distance between two embeddings.
# 
# #
class SiameseModel(nn.Module):
    def __init__(self, embedding_model, classification_model, decision_function=0):
        super(SiameseModel, self).__init__()
        self.embedding_model = embedding_model
        self.classification_model = classification_model
        self.decision_function = decision_function

        #define the distance calculation
        if self.decision_function == 0:
            self.distance = lambda a,b : torch.abs(torch.sub(a, b))
        elif self.decision_function == 1:
            self.d = torch.nn.PairwiseDistance(p=2)
            self.distance = lambda a,b : self.d(a,b).unsqueeze(1)
        elif self.decision_function == 2:
            self.distance = lambda a,b : torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-08).unsqueeze(1)
        else:
            self.distance = lambda a,b : torch.sub(a, b)


    def forward(self, a, b):
        a = self.embedding_model(a)
        b = self.embedding_model(b)
        c = self.distance(a, b)
        return self.classification_model(c)