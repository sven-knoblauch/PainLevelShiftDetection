import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader


class EmbeddingsModel(nn.Module):

    def __init__(self):
        super(EmbeddingsModel, self).__init__()

        self.embedding_model = nn.Sequential(
            nn.Linear(207, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

    def forward(self, a):
        return self.embedding_model(a)



class PainDataset(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.filter = filter
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


#"D:\Workspace\workspace_masterarbeit\FeatureGeneration\dataset_processed\\normalized_subjects.pkl"
class Trainer():
    def __init__(self, subjects_train, subjects_valid, path, valid_data_path=None, filter=None):
        self.path = path
        self.dataset1 = PainDataset(self.path, subjects=subjects_train, filter=filter)
        self.train_dataloader = DataLoader(self.dataset1, batch_size=128, shuffle=True)

        if valid_data_path:
            self.dataset2 = PainDataset(valid_data_path, subjects=subjects_valid, filter=filter)
        else:
            self.dataset2 = PainDataset(self.path, subjects=subjects_valid, filter=filter)
        self.valid_dataloader = DataLoader(self.dataset2, batch_size=128, shuffle=True)


        self.model = EmbeddingsModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.loss_func = losses.TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all")
        self.miner = miners.TripletMarginMiner(margin=0.02)

        self.verbose = 0
        self.epoch_train_loss = []
        self.epoch_valid_loss = []

    def train(self, EPOCHS):
        if self.verbose:
            print("started training")

        for epoch in range(EPOCHS):
            
            #training
            train_loss = []
            self.model.train()
            for data, labels in self.train_dataloader:
                self.optimizer.zero_grad()

                data, labels = data.to(self.device), labels.to(self.device)

                embeddings = self.model(data)
                hard_pairs = self.miner(embeddings, labels)
                loss = self.loss_func(embeddings, labels, hard_pairs)

                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.data)

            self.lr_scheduler.step()

            #validation
            valid_loss = []
            self.model.eval()
            for data, labels in self.valid_dataloader:
                data, labels = data.to(self.device), labels.to(self.device)

                embeddings = self.model(data)
                loss = self.loss_func(embeddings, labels)
                valid_loss.append(loss.data)

            self.epoch_train_loss.append(np.mean(train_loss))
            self.epoch_valid_loss.append(np.mean(valid_loss))
            
            if self.verbose:
                if epoch%1 == 0:
                    print("-----------------------")
                    print("epoch:{:3d} /{:2d}".format(epoch+1, EPOCHS))
                    print("train loss:  {:4.2f}".format(np.mean(train_loss)))
                    print("test loss:   {:4.2f}".format(np.mean(valid_loss)))


    def calculate_valid_acc(self, use_embeddings=True, max_depth=20):
        #prep train data
        train_data = torch.tensor(self.dataset1.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)
        if use_embeddings:
            train_data = self.model(train_data).detach()
        train_data = train_data.numpy()
        train_label = self.dataset1.data["pain"]
        
        #prep test data
        test_data = torch.tensor(self.dataset2.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)
        if use_embeddings:
            test_data = self.model(test_data).detach()
        test_data = test_data.numpy()
        test_label = self.dataset2.data["pain"]

        #train forest and predict on test data
        clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
        clf.fit(train_data, train_label)
        prediction = clf.predict(test_data)

        return np.sum(prediction == test_label)/len(test_label)


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

    def display_embeddings(self, use_model=True):
        data = torch.tensor(self.dataset2.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)
        if use_model:
            data = self.model(data)
        label = self.dataset2.data["pain"]
        self.plot_embeddings(data.detach().numpy(), label)




















"""
class EmbeddingsModel(nn.Module):

    def __init__(self):
        super(EmbeddingsModel, self).__init__()

        self.embedding_model = nn.Sequential(
            nn.Linear(207, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    def forward(self, a):
        return self.embedding_model(a)
"""
