import os
import ast
import csv
import math
import torch
import numpy as np
import settings
from utils import *
from settings import *

class InputExample(object):
  """Constructs an InputExample
  
  Args:
    text_a: 2-D list. Untokenized sentences of sequence a.
    text_b: 2-D list. Untokenized sentences of sequence b.
    labels: 2-D list. One-hot labels correspond to
            each sentence in context.
  """

  def __init__(self, text_a, text_b, labels=None):
    self.text_a = text_a
    self.text_b = text_b
    self.labels = labels

class DataReader(object):
  """
  Base class for data converters for sequence classification
  data sets.
  """

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class MRPCReader(DataReader):
  def __init__(self, batch_size=BATCH_SIZE):
    super().__init__()
    self.batch_size = batch_size

  def get_train_examples(self, data_dir):
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') 

  def get_dev_examples(self, data_dir):
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

  def get_test_examples(self, data_dir):
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, 'test.tsv')), 'test')

  def get_labels(self):
    return {'0': 0, '1': 1}

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    total_examples = len(lines)

    text_a = []
    text_b = []
    labels = []
    print("\rProcessed Examples: {}/{}".format(0,
                                          total_examples),
          end='\r', file=settings.SHELL_OUT_FILE, flush=True)
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      if i % 1000 == 0:
        print("\rProcessed Examples: {}/{}".format(i, total_examples),
              end='\r', file=settings.SHELL_OUT_FILE, flush=True)
      text_a.append(convert_to_unicode(line[3]))
      text_b.append(convert_to_unicode(line[4]))
      if set_type == "test":
        label = '0'
      else:
        label = convert_to_unicode(line[0])
      labels.append(label)

      if i % self.batch_size == 0:
        examples.append(
          InputExample(text_a=text_a, text_b=text_b, labels=labels))
        text_a = []
        text_b = []
        labels = []
    if len(text_a):
      examples.append(
        InputExample(text_a=text_a, text_b=text_b, labels=labels))
    print("\rProcessed Examples: {}/{}".format(total_examples,
                                               total_examples),
          file=settings.SHELL_OUT_FILE, flush=True)
    return examples

class MRPCProcessor(object):
  def __init__(self, tokenizer, max_seq_len=MAX_SEQ_LEN):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len

  def convert_examples_to_tensor(self, examples):
    # init
    length = len(examples.text_a)
    inputs_ids = np.zeros((length, self.max_seq_len), dtype=np.int64)
    token_type_ids = np.zeros_like(inputs_ids)

    for i, (text_a, text_b) in enumerate(zip(examples.text_a,
                                             examples.text_b)):
      tokens = []
      segment_ids = []
      tokens_a = self.tokenizer.tokenize(text_a)
      tokens_b = self.tokenizer.tokenize(text_b)
      truncate_seq_pair(tokens_a, tokens_b, self.max_seq_len - 3)
      # inputs
      tokens.extend(['[CLS]'] + tokens_a + ['[SEP]'])
      segment_ids.extend([0] * (len(tokens_a) + 2))
      tokens.extend(tokens_b + ['[SEP]'])
      segment_ids.extend([1] * (len(tokens_b) + 1))
      # pad
      pad_len = self.max_seq_len - len(tokens)
      tokens.extend(['[PAD]'] * pad_len)
      segment_ids.extend([0] * pad_len)
      # convert to ids
      tokens = self.tokenizer.convert_tokens_to_ids(tokens)
      # to numpy
      inputs_ids[i, :] = tokens[:]
      token_type_ids[i, :] = segment_ids[:]
    # to tensor
    inputs_ids = torch.from_numpy(inputs_ids).long().detach()
    token_type_ids = torch.from_numpy(token_type_ids).long().detach()
    inputs_mask = (inputs_ids != 0).long().detach()
    inputs = [inputs_ids, token_type_ids, inputs_mask]

    if settings.USE_CUDA:
      inputs = [i.cuda() for i in inputs]

    return inputs

  def convert_labels_to_tensor(self, labels, labels_dict):
    labels = np.array([labels_dict[t] for t in labels])
    labels = torch.from_numpy(labels).long()
    labels = labels.cuda() if settings.USE_CUDA else labesls
    return labels.detach()

