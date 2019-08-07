# Overview

The framework is built based on PyTorch framework, mainly used on training and testing model. Evaluation runs based on step when achieving specific rules.

# Usage

Put data in corresponding folders and modify codes in src/ folder.

- **src/**: all codes. Read README.md in the folder.
- **results/**: save all results in the folder, each run will built an independent folder and all files related to current run will be save in the folder.
- **data/**:
	- **bert-embedding/**: put all BERT model file in the folder. It can be change to any pretrained model (Please change script and codes correspondly).
	- **MRPC/**: put all data in the folder. Recommand name rule: 'train.txt' for training set file, 'dev.txt' for dev set file, 'test.txt' for test set file, and 'ori/' folder for original raw data.
- **run-single.sh**: script to train and test model.

