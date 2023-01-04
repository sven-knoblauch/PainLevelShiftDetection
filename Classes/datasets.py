import pandas as pd
from torch.utils.data import Dataset
import torch
from itertools import combinations


# #
# 
# Dataset for pain samples with label (pain or no pain)
# 
# #
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


# #
# 
# Dataset for siamese data, results are an anchor with a positive and negative sample, where the first two samples have the same label (pos)
# These triplets have the same subject
# 
# #
class SiameseDataset(Dataset):
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

        subj = sample["subject"][0]
        subj_data = self.data[self.data["subject"]==subj]
        pain = subj_data[subj_data["pain"]==1].sample()
        no_pain = subj_data[subj_data["pain"]==0].sample()

        #convert
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        no_pain = torch.tensor(no_pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        pain = torch.tensor(pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        if label == 0:
            return sample, no_pain, pain

        return sample, pain, no_pain


# #
# 
# Dataset for siamese data, results are an anchor with a positive and negative sample, where the first two samples have the same label (pos)
# These triplets have NOT the same subject
# 
# #
class SiameseDataset2(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        if filter:
            self.data = self.data[filter(self.data)]

        self.pain = self.data[self.data["pain"]==1]
        self.no_pain = self.data[self.data["pain"]==0]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[[idx]].reset_index(drop=True)
        label = sample["pain"][0]
        pain = self.pain.sample()
        no_pain = self.no_pain.sample()

        #convert
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        no_pain = torch.tensor(no_pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        pain = torch.tensor(pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        if label == 0:
            return sample, no_pain, pain

        return sample, pain, no_pain

# #
# 
# Dataset for siamese data, results are an anchor with a positive and negative sample, the positive sample is defined with the last output value
# which describes the label. Zero if the 
#
# #
class SiameseDatasetWithLabels(Dataset):
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

        subj = sample["subject"][0]
        subj_data = self.data[self.data["subject"]==subj]

        pain = subj_data[subj_data["pain"]==1].sample()
        no_pain = subj_data[subj_data["pain"]==0].sample()

        #convert
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        no_pain = torch.tensor(no_pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        pain = torch.tensor(pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        if label == 0:
            return sample, no_pain, pain, torch.tensor(0)

        return sample, pain, no_pain, torch.tensor(1)


class SiameseDatasetWithLabels2(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        if filter:
            self.data = self.data[filter(self.data)]

        self.pain = self.data[self.data["pain"]==1]
        self.no_pain = self.data[self.data["pain"]==0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[[idx]].reset_index(drop=True)
        label = sample["pain"][0]
        
        pain = self.pain.sample()
        no_pain = self.no_pain.sample()

        #convert
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        no_pain = torch.tensor(no_pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        pain = torch.tensor(pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        if label == 0:
            return sample, no_pain, pain, torch.tensor(0)

        return sample, pain, no_pain, torch.tensor(1)


# #
# 
# Dataset for siamese data, the triplet consists of two datapoints with the corresponding label. The label is zero if both samples have the same
# label, for example pain. Otherwise the returning label is one.
# 
# #
class SiameseDatasetCombinations(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        if filter:
            self.data = self.data[filter(self.data)].reset_index(drop=True)

        self.indices = []

        for sub in self.subjects:
            tmp = self.data[self.data["subject"] == sub].index
            self.indices.extend(list(combinations(tmp, 2)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx]

        sample1 = self.data.iloc[[indices[0]]].reset_index(drop=True)
        sample2 = self.data.iloc[[indices[1]]].reset_index(drop=True)

        label1 = sample1["pain"][0]
        label2 = sample2["pain"][0]

        #convert
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        label = torch.tensor(1-label1==label2, dtype=torch.float32)

        return sample1, sample2, label