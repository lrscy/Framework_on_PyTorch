# Source Code of the framework

## Overview

The framework now is built based on MRPC task using BERT on PyTorch framework.
Some of codes are modified from ['run_classifier.py'](https://github.com/google-research/bert/blob/master/run_classifier.py) file
in [BERT](https://github.com/google-research/bert) source code.

## Usage

Replace codes you need. Most common codes that need to be changed in main.py are comment out.

Usage of each file:
1. data_processor.py: Read your own file; convert examples and labels to tensor. Sometimes labels may get from data, delete functions you don't need.
2. settings.py: Default values.
3. parser.py: Parse parameters in shell.
4. models.py: All model's classes.
5. main.py: The core of the framework, concatenate all above and provide log info.
6. utils: Functions which are common used and may be used anywhere in framework.
