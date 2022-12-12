import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import wandb
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


class EmbeddingsModel(nn.Module):

    def __init__(self, model_hyperparameter):
        super(EmbeddingsModel, self).__init__()
        self.model_hyperparameter = model_hyperparameter
        self.sigmoid = self.model_hyperparameter["sigmoid"]
        self.dropout = self.model_hyperparameter["dropout"]
        self.fc1 = self.model_hyperparameter["fc1"]
        self.fc2 = self.model_hyperparameter["fc2"]
        self.fc3 = self.model_hyperparameter["fc3"]

        self.embedding_model = nn.Sequential(
            nn.Linear(207, self.fc1),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc1),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc1, self.fc2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc2),
            nn.Linear(self.fc2, self.fc3)
        )

        if self.sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        

    def forward(self, a):
        a = self.embedding_model(a)
        return self.activation(a)
        
            




class ClassificationModel(nn.Module):

    def __init__(self):
        super(ClassificationModel, self).__init__()

        self.classification_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        return self.classification_head(torch.abs(a-b))







class EmbeddingTrainer():
    def __init__(self, hyperparameters, model, device="cpu"):
        self.hyperparameters = hyperparameters
        
        #parameters
        self.path_train = self.hyperparameters["path_train"]
        self.path_test = self.hyperparameters["path_test"]
        self.subjects_train = self.hyperparameters["subjects_train"]
        self.subjects_test = self.hyperparameters["subjects_test"]
        self.wandb = self.hyperparameters["wandb"]
        self.acc_in_loop = self.hyperparameters["acc_in_loop"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.margin = self.hyperparameters["margin"]
        self.filter = self.hyperparameters["filter"]

        if self.hyperparameters["distance"] == 0:
            self.distance = distances.CosineSimilarity()
        else:
            self.distance = distances.LpDistance()


        #data
        self.train_dataset = PainDataset(self.path_train, subjects=self.subjects_train, filter=self.filter)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = PainDataset(self.path_test, subjects=self.subjects_test, filter=self.filter)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.datasets_dic = {"train": self.train_dataset, "test": self.test_dataset}



        #model
        self.device = torch.device(device)
        self.model = model.to(self.device)

        #learning objectives
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.distance = self.distance
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(margin=self.margin, distance=self.distance, reducer=self.reducer, triplets_per_anchor="all")
        self.mining_func = miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="all")
        self.accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)


        #testing
        self.tester = testers.GlobalEmbeddingSpaceTester(accuracy_calculator=self.accuracy_calculator, data_device=self.device)


    def get_all_embeddings(self, dataset):
        return self.tester.get_all_embeddings(dataset, self.model)


    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            embeddings = self.model(data)
            indices_tuple = self.mining_func(embeddings, labels)
            loss = self.loss_func(embeddings, labels, indices_tuple)
            loss.backward()
            self.optimizer.step()


    def test(self):
        return self.tester.test(self.datasets_dic, 0, self.model, embedder_model=None, splits_to_eval=None, collate_fn=None)


    def trainloop(self, epochs):
        history = []
        for epoch in range(1, epochs+1):
            self.train(epoch)
            tmp = self.test()
            history.append(tmp)

            if self.wandb:
                acc = self.test_accuracy()
                wandb.log({"accuracy": acc, "epoch": epoch})

            if self.acc_in_loop:
                acc = self.test_accuracy()
                print(acc)

        return history


    def plot_history(self, history):
        train_hist = [x["train"]["precision_at_1_level0"] for x in history]
        test_hist = [x["test"]["precision_at_1_level0"] for x in history]

        plt.plot(train_hist, label="train")
        plt.plot(test_hist, label="test")
        plt.legend()
        plt.show()


    def plot_embeddings(self, embeddings, labels):
        lower_embeddings = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(embeddings)

        df = pd.DataFrame([])
        df['tsne-2d-one'] = lower_embeddings[:,0]
        df['tsne-2d-two'] = lower_embeddings[:,1]
        df['y'] = labels

        plt.figure(figsize=(8,5))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df,
            legend="full",
            alpha=0.6
        )

    def display_embeddings(self, use_model=True, use_pain_class_label=False):
        data = torch.tensor(self.test_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        if use_model:
            data = self.model(data)
        if use_pain_class_label:
            label = self.test_dataset.data["label"]
            mask = self.test_dataset.data["pain"]
            label = label*mask
        else:
            label = self.test_dataset.data["pain"]
        self.plot_embeddings(data.cpu().detach().numpy(), label)

    def test_accuracy(self,max_depth=20):
        data_test = torch.tensor(self.test_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        data_test = self.model(data_test).cpu().detach().numpy()
        label_test = self.test_dataset.data["pain"]

        data_train = torch.tensor(self.train_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        data_train = self.model(data_train).cpu().detach().numpy()
        label_train = self.train_dataset.data["pain"]

        #train random forest on train data and evaluate on test data
        clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
        clf.fit(data_train, label_train)
        prediction = clf.predict(data_test)
        return np.sum(prediction == label_test)/len(label_test)






class PainDataset(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        
        if filter:
            self.data = self.data[filter(self.data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[[idx]].reset_index(drop=True)
        label = sample["pain"][0]

        #convert
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        label = torch.tensor(label, dtype=torch.float32)

        return sample, label










"""

class EmbeddingsModel(nn.Module):

    def __init__(self):
        super(EmbeddingsModel, self).__init__()

        self.embedding_model = nn.Sequential(
            nn.Linear(207, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    def forward(self, a):
        return self.embedding_model(a)

"""