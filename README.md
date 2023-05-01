# PainLevelShiftDetection
Pain level shift detection on the xite dataset and transfer learning to real medical dataset from intensive care patients.

## Introduction
This repository has the corresponding code for my master thesis in computer science. The idea is to classify two pain samples at the same time and test if they have the same pain level or if they have different pain levels. These results will be later transfered to real world medical data with the premise to recognize if pain medication did work and the pain level has shifted or the pain level didn't change and the pain drug didn't work.

## Structure
In the following paragraph, the different folders and their contents are described.

- **Classes:** Containing the classes for the neuronal networks of the siamese network and the related classes

- **DatasetExploration:** Containing jupyter notebooks for inspecting the generated dataset/features

- **EmbeddingsGenerator:** Jupyter notebooks for generating the emebdding network and testing the embedding network

- **FeatureGeneration:** All files related to the generation of the feature vectors, extracted from the biometric signals

- **PainCLassifivationTest:** Jupyter notebook for testing the generated feature vectors in a pain classification setting

- **RandomForestDatasetGenerator:** Generation of a dataset for a random forest, with embedding vectors after the distance calculation

- **SiameseNetwork:** Jupyter notebooks for training different siamese networks

- **ThreeClassSiamese:** Jupyter notebooks for the testing of the adjsuted asymetric siamese network

- **TransferLearning:** Jupyter notebook for testing transfer learning in the symmetric siamese network setting