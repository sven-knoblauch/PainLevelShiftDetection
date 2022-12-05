import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd





class PainDataset(Dataset):
    def __init__(self, path, subjects=["S001"], transform=None):
        self.path = path
        self.transform = transform
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.data = self.data[self.data["subject"].isin(self.subjects)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = self.data.iloc[[idx]]
        subject = sample["subject"].iloc[0]
        subject_mask = self.data['subject'] == subject
        subject_data = self.data[subject_mask]

        pain_data_mask = subject_data['pain'] == 1
        pain_data_sample = subject_data[pain_data_mask].sample()
        nopain_data_mask = subject_data['pain'] == 0
        nopain_data_sample = subject_data[nopain_data_mask].sample()


        sample_pain_order = sample["pain"].iloc[0]

        #make tensors
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1).values, dtype=torch.float32)[0]
        nopain_data_sample = torch.tensor(nopain_data_sample.drop(["pain", "subject", "label"], axis=1).values, dtype=torch.float32)[0]
        pain_data_sample = torch.tensor(pain_data_sample.drop(["pain", "subject", "label"], axis=1).values, dtype=torch.float32)[0]

        if self.transform:
            sample = self.transform(sample)


        if sample_pain_order == 1:
            return sample, pain_data_sample, nopain_data_sample
        
        return sample, nopain_data_sample, pain_data_sample


class SiameseModel(nn.Module):

    def __init__(self):
        super(SiameseModel, self).__init__()

        self.embedding_model = nn.Sequential(
            nn.Linear(207, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.estimator = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.distance = lambda a, b : torch.abs(a-b)

    def forward(self, a, b):
        a = self.embedding_model(a)
        b = self.embedding_model(b)
        d = self.distance(a, b)
        
        return self.estimator(d)






class PainDatasetValidation(Dataset):
    def __init__(self, path, subject="S001", transform=None):
        self.path = path
        self.data = pd.read_pickle(path)
        self.subject = subject
        self.data = self.data[self.data["subject"] == self.subject]

        self.df_anchor = pd.DataFrame([])
        self.df_sample = pd.DataFrame([])
        for index in range(len(self.data)-1):
            samples = self.data.iloc[index+1:]
            anchor = pd.concat([self.data.iloc[index]]*len(samples), axis=1, ignore_index=True).T
            self.df_anchor = pd.concat([self.df_anchor, samples], ignore_index=True)
            self.df_sample = pd.concat([self.df_sample, anchor], ignore_index=True)

        self.label = self.df_anchor["pain"] != self.df_sample["pain"]
        self.df_anchor = self.df_anchor.drop(["pain", "subject", "label"], axis=1)
        self.df_sample = self.df_sample.drop(["pain", "subject", "label"], axis=1)

    def __len__(self):
        return len(self.df_anchor)

    def __getitem__(self, idx):
        a = torch.tensor(self.df_anchor.iloc[idx].array, dtype=torch.float32)
        b = torch.tensor(self.df_sample.iloc[idx].array, dtype=torch.float32)
        c = torch.tensor(self.label[idx], dtype=torch.float32)

        return a, b, c




class PainDatasetTraining(Dataset):
    def __init__(self, path, subjects=["S001"], transform=None):
        self.path = path
        self.transform = transform
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.data = self.data[self.data["subject"].isin(self.subjects)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data.iloc[[idx]].reset_index(drop=True)
        subject = anchor["subject"].iloc[0]
        subject_data = self.data[self.data['subject'] == subject]
        sample = subject_data.sample().reset_index(drop=True)
        
        #calculate label
        label = anchor["pain"] != sample["pain"]

        #convert
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1).values, dtype=torch.float32)[0]
        anchor = torch.tensor(anchor.drop(["pain", "subject", "label"], axis=1).values, dtype=torch.float32)[0]
        label = torch.tensor(label, dtype=torch.float32)

        return anchor, sample, label