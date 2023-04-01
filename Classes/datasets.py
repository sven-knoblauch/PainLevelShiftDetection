import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from itertools import combinations


# #
# 
# Dataset for pain samples with label (pain or no pain)
# 
# #
class PainDataset(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample a datapoint and load his pain label
        sample = self.data.iloc[[idx]].reset_index(drop=True)
        label = sample["pain"][0]

        #convert to torch tensors and remove all columns, which arent features
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        label = torch.tensor(label, dtype=torch.float32)

        return sample, label

# #
# 
# Dataset for siamese data, results are an anchor with a positive and negative sample, the positive sample is defined with the last output value
# which describes the label. Zero if the first two samples have the same label (both pain or no pain), one otherwise.
#
# #
class SiameseDatasetWithLabels(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)]

        self.n = len(self.data)


    def __len__(self):
        return self.n*4

    def __getitem__(self, idx):
        idx = idx % self.n
        #sample a datapoint and load his pain label
        sample = self.data.iloc[[idx]].reset_index(drop=True)
        label = sample["pain"][0]

        #load data from same subject   
        subj = sample["subject"][0]
        subj_data = self.data[self.data["subject"]==subj]

        #sample pos and neg sample from same sibject
        pain = subj_data[subj_data["pain"]==1].sample()
        no_pain = subj_data[subj_data["pain"]==0].sample()

        #convert to torch tensors and remove all columns, which arent features
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        no_pain = torch.tensor(no_pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        pain = torch.tensor(pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        # return the correct label for positive and negative sample
        if label == 0:
            return sample, no_pain, pain, label
        return sample, pain, no_pain, label


# #
# 
# same as SiameseDatasetWithLabels, but with the change that the samples are not restricted to the same subject.
#
# #
class SiameseDatasetWithLabelsIgnoredSampleSubject(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)]

        self.pain = self.data[self.data["pain"]==1]
        self.no_pain = self.data[self.data["pain"]==0]
        self.n = len(self.data)

    def __len__(self):
        return self.n*4

    def __getitem__(self, idx):
        idx = idx % self.n
        #sample a datapoint and load his pain label
        sample = self.data.iloc[[idx]].reset_index(drop=True)
        label = sample["pain"][0]
        
        #sample pain and no pain data, no matter the subject
        pain = self.pain.sample()
        no_pain = self.no_pain.sample()

        #convert to torch tensors and remove all columns, which arent features
        sample = torch.tensor(sample.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        no_pain = torch.tensor(no_pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        pain = torch.tensor(pain.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        # return the correct label for positive and negative sample
        if label == 0:
            return sample, no_pain, pain, label
        return sample, pain, no_pain, label




# #
# 
# Dataset for siamese data, the triplet consists of two datapoints with the corresponding label. The label is zero if both samples have the same
# label, for example pain. Otherwise the returning label is one.
# 
# #
class SiameseDatasetCombinations(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)].reset_index(drop=True)

        #calculate indices with combinations of size 2 for each subject. and save these indices.
        self.indices = []
        for sub in self.subjects:
            tmp = self.data[self.data["subject"] == sub].index
            self.indices.extend(list(combinations(tmp, 2)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #get the indices of two samples with the index on the combination list
        indices = self.indices[idx]

        #load both samples
        sample1 = self.data.iloc[[indices[0]]].reset_index(drop=True)
        sample2 = self.data.iloc[[indices[1]]].reset_index(drop=True)

        #get the labels
        label1 = sample1["pain"][0]
        label2 = sample2["pain"][0]

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        #calculate label, 0 when both have the same pain label, 1 otherwise      
        label = torch.tensor(1-(label1==label2), dtype=torch.float32)

        return sample1, sample2, label



# #
# 
# Dataset for siamese data, the triplet consists of two datapoints with the corresponding label. The label is zero if both samples have the same
# label, for example pain. Otherwise the returning label is one. (Ignoring subjects)
# 
# #
class SiameseDatasetCombinationsIgnoredSampleSubject(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)*10

    def __getitem__(self, idx):
        idx = idx % len(self.data)

        #load both samples
        sample1 = self.data.iloc[[idx]].reset_index(drop=True)
        sample2 = self.data.sample()
        
        #get the labels
        label1 = sample1["pain"].values[0]
        label2 = sample2["pain"].values[0]

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        #calculate label, 0 when both have the same pain label, 1 otherwise      
        label = torch.tensor(1-(label1==label2), dtype=torch.float32)

        return sample1, sample2, label



# #
# 
# Dataset for siamese data, the triplet consists of two datapoints with the corresponding label. The label is zero if both samples have the same
# label, for example pain. Otherwise the returning label is one.
# 
# #
class SiameseDatasetCombinationsWithPainLevel(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)].reset_index(drop=True)

        #calculate indices with combinations of size 2 for each subject. and save these indices.
        self.indices = []
        for sub in self.subjects:
            tmp = self.data[self.data["subject"] == sub].index
            self.indices.extend(list(combinations(tmp, 2)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #get the indices of two samples with the index on the combination list
        indices = self.indices[idx]

        #load both samples
        sample1 = self.data.iloc[[indices[0]]].reset_index(drop=True)
        sample2 = self.data.iloc[[indices[1]]].reset_index(drop=True)

        #get the labels
        label1 = sample1["pain"][0]
        label2 = sample2["pain"][0]

        #get pain levels
        pain_level1 = torch.tensor(sample1["label"][0])
        pain_level2 = torch.tensor(sample2["label"][0])

        if label1==0:
            pain_level1 = torch.tensor(0)
        if label2==0:
            pain_level2 = torch.tensor(0)

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        #calculate label, 0 when both have the same pain label, 1 otherwise      
        label = torch.tensor(1-(label1==label2), dtype=torch.float32)

        return sample1, sample2, label, pain_level1, pain_level2



# #
# 
# Dataset for siamese data, the triplet consists of two datapoints with the corresponding label. The label is zero if both samples have the same
# label, for example pain. Otherwise the returning label is one. (Ignoring subjects)
# 
# #
class SiameseDatasetCombinationsIgnoredSampleSubjectWithPainLevel(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load both samples
        sample1 = self.data.iloc[[idx]].reset_index(drop=True)
        sample2 = self.data.sample()
        
        #get the labels
        label1 = sample1["pain"].values[0]
        label2 = sample2["pain"].values[0]

        #get pain levels
        pain_level1 = torch.tensor(sample1["label"][0])
        pain_level2 = torch.tensor(sample2["label"][0])

        if label1==0:
            pain_level1 = torch.tensor(0)
        if label2==0:
            pain_level2 = torch.tensor(0)

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        #calculate label, 0 when both have the same pain label, 1 otherwise      
        label = torch.tensor(1-(label1==label2), dtype=torch.float32)

        return sample1, sample2, label, pain_level1, pain_level2





# #
# 
# Dataset for generating the data for training the random forest
#
# #
class CombinationDataset(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)].reset_index(drop=True)

        #calculate indices with combinations of size 2 for each subject. and save these indices.
        self.indices = []
        for sub in self.subjects:
            tmp = self.data[self.data["subject"] == sub].index
            self.indices.extend(list(combinations(tmp, 2)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #get the indices of two samples with the index on the combination list
        indices = self.indices[idx]

        #load both samples
        sample1 = self.data.iloc[[indices[0]]].reset_index(drop=True)
        sample2 = self.data.iloc[[indices[1]]].reset_index(drop=True)

        #get the labels
        label1 = sample1["pain"][0]
        label2 = sample2["pain"][0]

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        label1 = torch.tensor(label1, dtype=torch.float32)
        label2 = torch.tensor(label2, dtype=torch.float32)

        #calculate label, 0 when both have the same pain label, 1 otherwise      
        label = torch.tensor(1-(label1==label2), dtype=torch.float32)

        return sample1, sample2, label1, label2, label





# #
#
#  Datasets for Three class problem
#  
#   1 => pain + no pain
#   0 => pain + pain or no pain + no pain
#  -1 => no pain + pain
#
# #
class SiameseDatasetThreeClass(Dataset):
    def __init__(self, path, subjects=["S001"], filter=None, ignore_sample_subject=True):
        self.path = path
        #read data from file
        self.data = pd.read_pickle(path)
        self.subjects = subjects
        self.ignore_sample_subject = ignore_sample_subject
        #remove all dara which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        #filter data for example for only electric pain stimuli
        if filter:
            self.data = self.data[filter(self.data)]
        self.n = len(self.data)

    def __len__(self):
        return self.n*4

    def __getitem__(self, idx):
        idx = idx % self.n
        #sample a datapoint and load his pain label
        sample1 = self.data.iloc[[idx]].reset_index(drop=True)
        label1 = sample1["pain"][0]

        if self.ignore_sample_subject:
            sample2 = self.data.sample()
        else:
            #load data from same subject   
            subj = sample1["subject"][0]
            subj_data = self.data[self.data["subject"]==subj]
            sample2 = subj_data.sample()

        #get second label
        label2 = sample2["pain"].values[0]

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        # return the correct label for positive and negative sample
        label = torch.tensor((label1-label2), dtype=torch.float32)
        return sample1, sample2, label




class SiameseDatasetIntenseThreeClass(Dataset):
    def __init__(self, subjects=["1"], indices1=1, indices2=0):
        #read data from file
        self.data = pd.read_pickle("D:\Workspace\workspace_masterarbeit\PainLevelShiftDetection\FeatureGeneration\dataset_processed\INTENSE2\\normalized_subjects.pkl")
        self.subjects = subjects
        #remove all data which are not from the wanted subject list
        self.data = self.data[self.data["subject"].isin(self.subjects)].reset_index(drop=True)
        
        self.indices = []
        for x in range(len(self.data)//3):
            self.indices.append(np.array([1,0])+x*3)
            self.indices.append(np.array([1,2])+x*3)
        self.indices = [(x[0], x[1]) for x in self.indices]

        self.missing_indices = []
        for x in range(len(self.data)//3):
            self.missing_indices.append(np.array([2,2])+x*3)
            self.missing_indices.append(np.array([2,1])+x*3)
        self.missing_indices = [(x[0], x[1]) for x in self.missing_indices]

        self.indices = indices1*self.indices + indices2*self.missing_indices


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx]

        #load both samples
        sample1 = self.data.iloc[[indices[0]]].reset_index(drop=True)
        sample2 = self.data.iloc[[indices[1]]].reset_index(drop=True)

        #get the labels
        label1 = sample1["pain"][0]
        label2 = sample2["pain"][0]

        #convert to torch tensors and remove all columns, which arent features
        sample1 = torch.tensor(sample1.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]
        sample2 = torch.tensor(sample2.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32)[0]

        # return the correct label for positive and negative sample
        label = torch.tensor((label1-label2), dtype=torch.float32)
        return sample1, sample2, label