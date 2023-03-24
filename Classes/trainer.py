import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import wandb
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import PainDataset, SiameseDatasetWithLabels, SiameseDatasetCombinations, SiameseDatasetWithLabelsIgnoredSampleSubject
from datasets import SiameseDatasetCombinationsIgnoredSampleSubjectWithPainLevel, SiameseDatasetCombinationsIgnoredSampleSubject
from datasets import SiameseDatasetCombinationsWithPainLevel

from models import SiameseModel
from tqdm import tqdm
from IPython.display import clear_output


#all possible subjects
all_subjects = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008',
                'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016',
                'S017', 'S018', 'S019', 'S020', 'S021', 'S022', 'S023', 'S024',
                'S025', 'S026', 'S027', 'S029', 'S031', 'S032', 'S033', 'S034',
                'S035', 'S036', 'S037', 'S038', 'S039', 'S040', 'S041', 'S042',
                'S043', 'S044', 'S045', 'S046', 'S047', 'S048', 'S049', 'S050',
                'S051', 'S052', 'S053', 'S054', 'S055', 'S056', 'S057', 'S058',
                'S060', 'S061', 'S062', 'S063', 'S064', 'S065', 'S066', 'S067',
                'S068', 'S069', 'S070', 'S071', 'S072', 'S073', 'S074', 'S075',
                'S076', 'S077', 'S078', 'S079', 'S080', 'S081', 'S082', 'S083',
                'S084', 'S085', 'S086', 'S087', 'S088', 'S089', 'S090', 'S091',
                'S092', 'S093', 'S094', 'S095', 'S096', 'S097', 'S098', 'S099',
                'S100', 'S101', 'S102', 'S103', 'S104', 'S105', 'S106', 'S107',
                'S109', 'S110', 'S111', 'S112', 'S113', 'S114', 'S115', 'S116',
                'S117', 'S118', 'S119', 'S120', 'S121', 'S122', 'S123', 'S124',
                'S125', 'S126', 'S127', 'S128', 'S129', 'S130', 'S131', 'S132',
                'S133', 'S134']

all_subjects_intense = ['1', '10', '14', '15', '17', '19', '2', '20', '21', '23', '25',
                        '28', '29', '3', '30', '31', '32', '33', '35', '36', '37', '38',
                        '39', '6']

# #
# 
# Test Accuracy with random forest (with and without embedding model)
# 
# #
class AccuracyTester():
    def __init__(self, hyperparameters, filter=None, device="cpu"):
        self.hyperparameters = hyperparameters
        self.filter = filter
        self.device = device
        self.path = self.hyperparameters["path"]
        self.subjects_train = self.hyperparameters["subjects_train"]
        self.subjects_test = self.hyperparameters["subjects_test"]

        self.train_dataset = PainDataset(self.path, subjects=self.subjects_train, filter=self.filter)
        self.test_dataset = PainDataset(self.path, subjects=self.subjects_test, filter=self.filter)
        

    def test_model(self, max_depth=20, model=None):
        if model is not None:
            model.to(self.device)

        #define test data
        data_test = torch.tensor(self.test_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        if model is not None:
            data_test = model(data_test)
        data_test = data_test.cpu().detach().numpy()
        label_test = self.test_dataset.data["pain"]

        #define train data
        data_train = torch.tensor(self.train_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        if model is not None:
            data_train = model(data_train)
        data_train = data_train.cpu().detach().numpy()
        label_train = self.train_dataset.data["pain"]

        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=400, random_state=0, n_jobs=-1)
        clf.fit(data_train, label_train)
        prediction = clf.predict(data_test)
        return np.sum(prediction == label_test)/len(label_test)


# #
# 
# Trainer to train embedding model
# 
# #
class EmbeddingTrainer():
    def __init__(self, hyperparameters, model, filter=None, distance=None, device="cpu"):
        self.hyperparameters = hyperparameters
        
        #parameters
        self.path_train = self.hyperparameters["path_train"]
        self.path_test = self.hyperparameters["path_test"]
        self.subjects_train = self.hyperparameters["subjects_train"]
        self.subjects_test = self.hyperparameters["subjects_test"]
        self.acc_tester_metric = self.hyperparameters["acc_tester_metric"]
        self.wandb = self.hyperparameters["wandb"]
        self.acc_in_loop = self.hyperparameters["acc_in_loop"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.margin = self.hyperparameters["margin"]
        self.lr_steps = self.hyperparameters["lr_steps"]
        self.filter = filter

        if distance is None:
            self.distance = distances.LpDistance()
        else:
            self.distance = distance




        #data
        self.train_dataset = PainDataset(self.path_train, subjects=self.subjects_train, filter=self.filter)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_dataset = PainDataset(self.path_test, subjects=self.subjects_test, filter=self.filter)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)

        self.datasets_dic = {"train": self.train_dataset, "test": self.test_dataset}

        #model
        self.device = torch.device(device)
        self.model = model.to(self.device)

        #learning objectives
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_steps, gamma=0.5)

        self.distance = self.distance
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(margin=self.margin, distance=self.distance, reducer=self.reducer, triplets_per_anchor="all")
        self.mining_func = miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="all")
        self.accuracy_calculator = AccuracyCalculator(include=(self.acc_tester_metric,), k=1)

        #testing
        self.tester = testers.GlobalEmbeddingSpaceTester(accuracy_calculator=self.accuracy_calculator, data_device=self.device)
        self.history = []


    #get all embeddings calulcated by the model
    def get_all_embeddings(self, dataset):
        return self.tester.get_all_embeddings(dataset, self.model)

    #train for one epoch
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
        self.lr_scheduler.step()

    #test the model tih the tester on a predefined metric
    def test(self):
        return self.tester.test(self.datasets_dic, 0, self.model, embedder_model=None, splits_to_eval=None, collate_fn=None)

    #teh trainloop with training and testing. Log the history and add values to wandb if needed
    def trainloop(self, epochs):
        current_epoch = len(self.history)
        tmp = self.test()
        self.history.append(tmp)
        for epoch in range(1+current_epoch, epochs+1+current_epoch):
            self.train(epoch)
            tmp = self.test()
            self.history.append(tmp)

            if self.wandb:
                acc = self.test_accuracy()
                wandb.log({"accuracy": acc, "epoch": epoch})

            if self.acc_in_loop and epoch%2 == 0:
                acc = self.test_accuracy()
                print(acc)

    #display the history
    def plot_history(self):
        train_hist = [x["train"][self.acc_tester_metric+"_level0"] for x in self.history]
        test_hist = [x["test"][self.acc_tester_metric+"_level0"] for x in self.history]

        plt.plot(train_hist, label="train")
        plt.plot(test_hist, label="test")
        plt.legend()
        plt.show()

    #display the data with T-SNE
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

    #display embeddings with or withour the embedding model
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

    #test accuracy on a simple random forest
    def test_accuracy(self, max_depth=20):
        #define test data
        data_test = torch.tensor(self.test_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        data_test = self.model(data_test).cpu().detach().numpy()
        label_test = self.test_dataset.data["pain"]

        #define train data
        data_train = torch.tensor(self.train_dataset.data.drop(["pain", "subject", "label"], axis=1, errors='ignore').values, dtype=torch.float32).to(self.device)
        data_train = self.model(data_train).cpu().detach().numpy()
        label_train = self.train_dataset.data["pain"]

        #train random forest on train data and evaluate on test data
        clf = RandomForestClassifier(max_depth=max_depth, n_estimators=400, random_state=0,  n_jobs=-1)
        clf.fit(data_train, label_train)
        prediction = clf.predict(data_test)
        return np.sum(prediction == label_test)/len(label_test)








# #
# 
# Trainer to train Siamese model with a combination of the BCE loss and Triplet loss
# 
# #
class SiameseTrainerCombinedLoss():
    def __init__(self, hyperparameters, model_classifier, model_embedder, device="cpu", filter=None):
        self.hyperparameters = hyperparameters
        
        #parameters
        self.path = self.hyperparameters["path"]
        self.subjects_train = self.hyperparameters["subjects_train"]
        self.subjects_test = self.hyperparameters["subjects_test"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.margin = self.hyperparameters["margin"]
        self.batch_size_test = self.hyperparameters["batch_size_test"]
        self.number_steps = self.hyperparameters["number_steps"]
        self.number_steps_testing = self.hyperparameters["number_steps_testing"]
        self.lr_steps = self.hyperparameters["lr_steps"]
        self.wandb = self.hyperparameters["wandb"]
        self.log = self.hyperparameters["log"]
        self.dataset_ignore_sample_subject_train = self.hyperparameters["dataset_ignore_sample_subject_train"]
        self.dataset_ignore_sample_subject_test = self.hyperparameters["dataset_ignore_sample_subject_test"]
        self.filter = filter
        self.lambda_loss = self.hyperparameters["lambda_loss"]
        self.weight_decay = self.hyperparameters["weight_decay"]
        if self.weight_decay is None:
            self.weight_decay = 0

        #define datasets
        if self.dataset_ignore_sample_subject_train:
            self.train_dataset = SiameseDatasetWithLabelsIgnoredSampleSubject(self.path, subjects=self.subjects_train, filter=self.filter)
        else:
            self.train_dataset = SiameseDatasetWithLabels(self.path, subjects=self.subjects_train, filter=self.filter)

        if self.dataset_ignore_sample_subject_test:
            self.test_dataset = SiameseDatasetWithLabelsIgnoredSampleSubject(self.path, subjects=self.subjects_test, filter=self.filter)
        else:
            self.test_dataset = SiameseDatasetWithLabels(self.path, subjects=self.subjects_test, filter=self.filter)

        
        #define dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=True)

        #define dataset length
        if self.number_steps is None:
            self.number_steps = len(self.train_loader)
        else:
            self.number_steps = min(len(self.train_loader), self.number_steps)

        #define dataset length
        if self.number_steps_testing is None:
            self.number_steps_testing = len(self.test_loader)
        else:
            self.number_steps_testing = min(len(self.test_loader), self.number_steps_testing)

        #model
        self.device = torch.device(device)
        self.model_classifier = model_classifier.to(self.device)
        self.model_embedder = model_embedder.to(self.device)
        
        self.siamese_model = SiameseModel(self.model_embedder, self.model_classifier, decision_function=0)

        #optimizer
        self.optimizer = optim.Adam(self.siamese_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_steps, gamma=0.5)

        #combined loss
        self.distance = distances.LpDistance()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func_embedding = losses.TripletMarginLoss(margin=self.margin, distance=self.distance, reducer=self.reducer, triplets_per_anchor="hard").to(self.device)
        self.mining_func = miners.TripletMarginMiner(margin=self.margin, distance=self.distance, type_of_triplets="hard")

        self.loss_func_classification = nn.BCELoss().to(self.device)
        

        #logging
        self.history = []


    def train(self):
        self.model_embedder.train()
        self.model_classifier.train()
        self.siamese_model.train()
        history_loss = []
        history_acc = []

        #get mini batches
        for step, (anchor, pos, neg, label) in enumerate(tqdm(self.train_loader, total=self.number_steps, disable=not self.log)):
            anchor, pos, neg, label = anchor.to(self.device), pos.to(self.device), neg.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()

            #calculate embedding
            anchor = self.model_embedder(anchor)
            pos = self.model_embedder(pos)
            neg = self.model_embedder(neg)

            #embedding loss
            labels = torch.cat([label, label, 1-label])
            embeddings = torch.cat([anchor, pos, neg])
            indices_tuple = self.mining_func(embeddings, labels)
            loss_embedding = self.loss_func_embedding(embeddings, labels, indices_tuple)

            #prediction            
            pred_equal = self.model_classifier(torch.abs(torch.sub(anchor, pos))).flatten()     #label=0
            pred_unequal = self.model_classifier(torch.abs(torch.sub(anchor, neg))).flatten()   #label=1

            #accuracy
            acc_ecual = torch.sum((pred_equal < 0.5))/len(pred_equal)
            acc_unecual = torch.sum((pred_unequal >= 0.5))/len(pred_unequal)

            #loss
            loss1 = self.loss_func_classification(pred_equal, torch.zeros(len(pred_equal)).to(self.device))
            loss2 = self.loss_func_classification(pred_unequal, torch.ones(len(pred_unequal)).to(self.device))
            loss_classification = 1/2*(loss1+loss2)
            
            #combine loss
            loss = (1-self.lambda_loss)*loss_classification + self.lambda_loss *loss_embedding

            #log history
            history_acc.append(acc_ecual)
            history_acc.append(acc_unecual)
            history_loss.append(loss.data)

            #backpropagation
            loss.backward()
            self.optimizer.step()

            if step >= self.number_steps:
                break

        self.lr_scheduler.step()

        return {"loss": torch.tensor(history_loss).mean(), "acc": torch.tensor(history_acc).mean()}

    def test(self):
        self.model_embedder.eval()
        self.model_classifier.eval()
        self.siamese_model.eval()
        history_loss = []
        history_acc = []

        #get mini batches
        for step, (anchor, pos, neg, label) in enumerate(tqdm(self.test_loader, total=self.number_steps_testing, disable=not self.log)):
            anchor, pos, neg, label = anchor.to(self.device), pos.to(self.device), neg.to(self.device), label.to(self.device)

            #calculate embedding
            anchor = self.model_embedder(anchor)
            pos = self.model_embedder(pos)
            neg = self.model_embedder(neg)

            #embedding loss
            labels = torch.cat([label, label, 1-label])
            embeddings = torch.cat([anchor, pos, neg])
            indices_tuple = self.mining_func(embeddings, labels)
            loss_embedding = self.loss_func_embedding(embeddings, labels, indices_tuple)

            #prediction            
            pred_equal = self.model_classifier(torch.abs(torch.sub(anchor, pos))).flatten() #label=0
            pred_unequal = self.model_classifier(torch.abs(torch.sub(anchor, neg))).flatten() #label=1
            
            #accuracy calculation
            acc_ecual = torch.sum((pred_equal < 0.5))/len(pred_equal)
            acc_unecual = torch.sum((pred_unequal >= 0.5))/len(pred_unequal)

            #loss calculation
            loss1 = self.loss_func_classification(pred_equal, torch.zeros(len(pred_equal)).to(self.device))
            loss2 = self.loss_func_classification(pred_unequal, torch.ones(len(pred_unequal)).to(self.device))
            loss_classification = 1/2*(loss1+loss2)

            #combine loss
            loss = (1-self.lambda_loss)*loss_classification + self.lambda_loss *loss_embedding

            #log history
            history_acc.append(acc_ecual)
            history_acc.append(acc_unecual)
            history_loss.append(loss.data)

            if step >= self.number_steps_testing:
                break

        return {"loss": torch.tensor(history_loss).mean(), "acc": torch.tensor(history_acc).mean()}
    
    #the trainloop with the training and testing phase and tracking of the history
    def trainloop(self, epochs):
        current_epoch = len(self.history)
        for epoch in range(current_epoch+1, current_epoch+epochs+1):
            h_train = self.train()
            h_test = self.test()
            tmp = {"epoch":epoch, "train":h_train, "test":h_test}
            self.history.append(tmp)
            if self.log:
                clear_output(wait=True)
                for entry in self.history:
                    print(entry)
            if self.wandb:
                wandb.log({"accuracy": h_test["acc"], "epoch": epoch})

    #plot the history (loss and accuracy)
    def plot_history(self):
        #get data for plotting
        train_loss = [x["train"]["loss"] for x in self.history]
        test_loss = [x["test"]["loss"] for x in self.history]
        train_acc = [x["train"]["acc"] for x in self.history]
        test_acc = [x["test"]["acc"] for x in self.history]

        #plot loss
        plt.plot(train_loss, label="train")
        plt.plot(test_loss, label="test")
        plt.title("loss")
        plt.legend()
        plt.show()

        #plot accuracy
        plt.plot(train_acc, label="train")
        plt.plot(test_acc, label="test")
        plt.title("accuracy")
        plt.legend()
        plt.show()









# #
# 
# Trainer to train Siamese model
# 
# #
class SiameseTrainerCombinationDataset():
    def __init__(self, hyperparameters, siamese_model, device="cpu", filter=None):
        self.hyperparameters = hyperparameters
        
        #parameters
        self.path = self.hyperparameters["path"]
        self.subjects_train = self.hyperparameters["subjects_train"]
        self.subjects_test = self.hyperparameters["subjects_test"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.batch_size_test = self.hyperparameters["batch_size_test"]
        self.lr_steps = self.hyperparameters["lr_steps"]
        self.adam = self.hyperparameters["adam"]
        self.wandb = self.hyperparameters["wandb"]
        self.log = self.hyperparameters["log"]
        self.dataset_ignore_subject_train = self.hyperparameters["dataset_ignore_subject_train"]
        self.dataset_ignore_subject_test = self.hyperparameters["dataset_ignore_subject_test"]
        self.pain_levels =  self.hyperparameters["filter"]
        self.pain_levels = self.pain_levels + [0]
        self.pain_levels = np.sort(self.pain_levels)
        self.filter = filter
        self.number_steps = self.hyperparameters["number_steps"]
        self.number_steps_testing = self.hyperparameters["number_steps_testing"]
        self.number_steps_histogramm = self.hyperparameters["number_steps_histogramm"]
        self.weight_decay = self.hyperparameters["weight_decay"]
        if self.weight_decay is None:
            self.weight_decay = 0


        #data
        if self.dataset_ignore_subject_train:
            self.train_dataset = SiameseDatasetCombinationsIgnoredSampleSubject(self.path, subjects=self.subjects_train, filter=self.filter)
        else:
            self.train_dataset = SiameseDatasetCombinations(self.path, subjects=self.subjects_train, filter=self.filter)

        if self.dataset_ignore_subject_test:
            self.test_dataset = SiameseDatasetCombinationsIgnoredSampleSubject(self.path, subjects=self.subjects_test, filter=self.filter)
            self.test_dataset_with_pain_level = SiameseDatasetCombinationsIgnoredSampleSubjectWithPainLevel(self.path, subjects=self.subjects_test, filter=self.filter)
        else:
            self.test_dataset = SiameseDatasetCombinations(self.path, subjects=self.subjects_test, filter=self.filter)
            self.test_dataset_with_pain_level = SiameseDatasetCombinationsWithPainLevel(self.path, subjects=self.subjects_test, filter=self.filter)


        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=True)
        self.test_loader_with_pain_level = DataLoader(self.test_dataset_with_pain_level, batch_size=self.batch_size_test, shuffle=True)


        if self.number_steps is None:
            self.number_steps = len(self.train_loader)
        else:
            self.number_steps = min(len(self.train_loader), self.number_steps)

        if self.number_steps_testing is None:
            self.number_steps_testing = len(self.test_loader)
        else:
            self.number_steps_testing = min(len(self.test_loader), self.number_steps_testing)

        if self.number_steps_histogramm is None:
            self.number_steps_histogramm = len(self.test_loader_with_pain_level)
        else:
            self.number_steps_histogramm = min(len(self.test_loader_with_pain_level), self.number_steps_histogramm)

        #training loop
        self.loss_func = nn.BCELoss()

        #model
        self.device = torch.device(device)
        self.siamese_model = siamese_model.to(self.device)

        #optimizer
        if self.adam:
            self.optimizer = optim.Adam(self.siamese_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.siamese_model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_steps, gamma=0.2)

        #logging
        self.history = []
        self.history_cm = []

    def train(self):
        self.siamese_model.train()
        history_loss = []
        history_acc = []
        
        for step, (sample1, sample2, labels) in enumerate(tqdm(self.train_loader, total=self.number_steps, disable=not self.log)):
            sample1, sample2, labels = sample1.to(self.device), sample2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            #prediction            
            predictions = self.siamese_model(sample1, sample2).flatten()

            #accuracy
            acc = torch.sum((predictions >= 0.5) == labels)/len(labels)

            #loss
            loss = self.loss_func(predictions, labels)
            
            #log history
            history_acc.append(acc)
            history_loss.append(loss.data)

            #backpropagation
            loss.backward()
            self.optimizer.step()

            if step >= self.number_steps:
                break

        self.lr_scheduler.step()
        return {"loss": torch.tensor(history_loss).mean().item(), "acc": torch.tensor(history_acc).mean().item()}

    def test(self):
        self.siamese_model.eval()
        history_loss = []

        CM=0

        #get mini batches
        for step, (sample1, sample2, labels) in enumerate(tqdm(self.test_loader, total=self.number_steps_testing, disable=not self.log)):
            sample1, sample2, labels = sample1.to(self.device), sample2.to(self.device), labels.to(self.device)

            #prediction            
            predictions = self.siamese_model(sample1, sample2).flatten()
            
            class_predictions = (predictions >= 0.5)

            CM += confusion_matrix(labels.cpu(), class_predictions.cpu(), labels=[0,1])

            #loss
            loss = self.loss_func(predictions, labels)
            
            #log history
            history_loss.append(loss.data)

            if step >= self.number_steps_testing:
                break

        acc=np.sum(np.diag(CM)/np.sum(CM))

        return {"loss": torch.tensor(history_loss).mean().item(), "acc": acc, "cm": CM}

    #training loop with logging
    def trainloop(self, epochs):
        current_epoch = len(self.history)
        for epoch in range(1+current_epoch, epochs+current_epoch+1):
            h_train = self.train()
            h_test = self.test()
            tmp = {"epoch":epoch,
                   "train_acc":np.round_(h_train["acc"], decimals=4),
                   "train_loss":np.round_(h_train["loss"], decimals=4),
                   "test_acc":np.round_(h_test["acc"], decimals=4),
                   "test_loss":np.round_(h_test["loss"], decimals=4)}
            self.history.append(tmp)
            self.history_cm.append({"epoch":epoch, "cm":h_test["cm"]})
            if self.log:
                clear_output(wait=True)
                for entry in self.history:
                    print("epoch:", entry["epoch"], "| train_acc:", entry["train_acc"], "| test_acc:", entry["test_acc"])

            if self.wandb:
                wandb.log({"accuracy": h_test["acc"], "epoch": epoch})

    def trainloop_no_testing(self, epochs):
        for epoch in tqdm(range(epochs)):
            tmp = self.train()
            clear_output(wait=True)
        print(tmp["acc"])

    #plot history
    def plot_history(self):
        #get data for plotting
        train_loss = [x["train_loss"] for x in self.history]
        test_loss = [x["test_loss"] for x in self.history]
        train_acc = [x["train_acc"] for x in self.history]
        test_acc = [x["test_acc"] for x in self.history]

        #plot loss
        plt.plot(train_loss, label="train")
        plt.plot(test_loss, label="test")
        plt.title("loss")
        plt.legend()
        plt.show()

        #plot accuracy
        plt.plot(train_acc, label="train")
        plt.plot(test_acc, label="test")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def plot_cm(self, cm, normalize=True):
        if normalize:
            s = np.sum(cm, axis=1)
            cm = cm.astype('float64')
            cm[0] = cm[0]/s[0]
            cm[1] = cm[1]/s[1]

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["same pain level", "different pain level"])
        cm_display.plot(cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix", fontsize=16)
        plt.grid(False)
        plt.show()

    def calculate_f_scores(self, cm):
        tn = cm[0][0]
        tp = cm[1][1]
        fn = cm[1][0]
        fp = cm[0][1]

        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        f1=2*(precision*recall)/(precision+recall)

        return {"recall": precision, "precision": recall, "f1": f1}

    #positive means pain shift detected
    def test_with_pain_levels(self):
        self.siamese_model.eval()

        tp_history = torch.tensor([]).to(self.device)
        tn_history = torch.tensor([]).to(self.device)
        fp_history = torch.tensor([]).to(self.device)
        fn_history = torch.tensor([]).to(self.device)

        #get mini batches
        for step, (sample1, sample2, labels, painlvl1, painlvl2) in enumerate(tqdm(self.test_loader_with_pain_level, total=self.number_steps_histogramm, disable=not self.log)):
            sample1, sample2, labels, painlvl1, painlvl2 = sample1.to(self.device), sample2.to(self.device), labels.to(self.device), painlvl1.to(self.device), painlvl2.to(self.device)

            #prediction            
            predictions = self.siamese_model(sample1, sample2).flatten()
            
            class_predictions = (predictions >= 0.5)

            tmp1 = class_predictions == labels
            tmp2 = ~tmp1
            pos = (labels == 1)
            neg = ~pos
            tp = tmp1 & pos
            tn = tmp1 & neg
            fp = tmp2 & pos
            fn = tmp2 & neg

            tp_history = torch.cat([tp_history, painlvl1[tp], painlvl2[tp]])
            tn_history = torch.cat([tn_history, painlvl1[tn], painlvl2[tn]])
            fp_history = torch.cat([fp_history, painlvl1[fp], painlvl2[fp]])
            fn_history = torch.cat([fn_history, painlvl1[fn], painlvl2[fn]])

            if step >= self.number_steps_histogramm:
                break

        return {"tp": tp_history.cpu(), "tn": tn_history.cpu(), "fp": fp_history.cpu(), "fn": fn_history.cpu()}


    def display_histograms(self):
        #get data
        data = self.test_with_pain_levels()

        #calculate histogramms
        hist_tp = torch.histc(data["tp"], bins=len(self.pain_levels), min=min(self.pain_levels), max=max(self.pain_levels)).numpy()
        hist_tn = torch.histc(data["tn"], bins=len(self.pain_levels), min=min(self.pain_levels), max=max(self.pain_levels)).numpy()
        hist_fp = torch.histc(data["fp"], bins=len(self.pain_levels), min=min(self.pain_levels), max=max(self.pain_levels)).numpy()
        hist_fn = torch.histc(data["fn"], bins=len(self.pain_levels), min=min(self.pain_levels), max=max(self.pain_levels)).numpy()
        histogramms = list(map(list, zip(hist_tp, hist_tn, hist_fp, hist_fn)))

        #plot results
        fig, axes = plt.subplots(5, 3, figsize=(15, 12), tight_layout=True)
        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        fig.suptitle('Histograms', fontsize=16)

        for ax, feature, name in zip(axes.flatten(), histogramms, self.pain_levels):
            ax.bar([1,2,3,4], feature, edgecolor='#000000')
            ax.set_xticks([1,2,3,4], ["tp", "tn", "fp", "fn"])
            ax.set(title="class "+str(name))

        for axe in axes.flatten()[len(self.pain_levels):]:
            axe.remove()






class siameseNetworkEnsemble():
    def __init__(self, hyperparameters, siamese_models):

        #hyperparameters
        self.hyperparameters = hyperparameters
        self.subjects_test = self.hyperparameters["subjects_test"]
        self.batch_size_test = self.hyperparameters["batch_size_test"]
        self.number_steps_testing = self.hyperparameters["number_steps_testing"]
        self.device = self.hyperparameters["device"]
        self.device = torch.device(self.device)
        self.path = self.hyperparameters["path"]
        self.siamese_models = siamese_models

        #dataset
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=True)
        self.test_dataset = SiameseDatasetCombinations(self.path, subjects=self.subjects_test, filter=None)
        if self.number_steps_testing is None:
            self.number_steps_testing = len(self.test_loader)

        #training loop
        self.loss_func = nn.BCELoss()

    def test(self):
        self.siamese_model.eval()
        history_loss = []

        CM=0

        #get mini batches
        for step, (sample1, sample2, labels) in enumerate(tqdm(self.test_loader, total=self.number_steps_testing)):
            sample1, sample2, labels = sample1.to(self.device), sample2.to(self.device), labels.to(self.device)

            #prediction            
            predictions = self.siamese_model(sample1, sample2).flatten()



            #todo majority rule etc




            
            class_predictions = (predictions >= 0.5)

            CM += confusion_matrix(labels.cpu(), class_predictions.cpu(), labels=[0,1])

            #loss
            loss = self.loss_func(predictions, labels)
            
            #log history
            history_loss.append(loss.data)

            if step >= self.number_steps_testing:
                break

        acc=np.sum(np.diag(CM)/np.sum(CM))

        return {"loss": torch.tensor(history_loss).mean().item(), "acc": acc, "cm": CM}












