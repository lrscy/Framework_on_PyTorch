import os
import ast
import sys
import copy
import pprint
import settings
import torch.nn.functional as F
from settings import *
from utils import *
from parser import *
from models import *
from data_processor import *
from torch import optim
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer 
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import precision_recall_fscore_support

def make_model(args):
  ''' Making models here according to parameters in settings and args '''

  print('Making model...', file=settings.SHELL_OUT_FILE, flush=True)

  ### MODIFY START HERE ###
  c = copy.deepcopy
  bert_model = BertModel.from_pretrained(
                 args.bert_dir + args.bert_file + '.tar.gz')
  model = BertBase(bert_model, 2)
  model.init_parameters()

  optimizer = BertAdam(model.parameters(),
                       warmup=args.warmup_propotion,
                       lr=args.lr, weight_decay=args.l2)
  criterion = nn.CrossEntropyLoss()
  ### END HERE ###

  print('Done\n', file=settings.SHELL_OUT_FILE, flush=True)

  return model, optimizer, criterion

def save_best_model(model, prefix, name, total_step):
  file_name = prefix + name
  state = {'step': total_step + 1,
           'state_dict': model.state_dict()}
  torch.save(state, file_name)

def evaluate(criterion, output, label, outputs, labels):
  '''
  Compute loss and record results here.
  Designed for dev set and test set.

  Args:
    criterion: Loss function of your model.
    output: output by your model.
    label: golden label of your output data respectively.
    outputs: a list of outputs saving all outputs.
    labels: a list of golden labels of outputs respectively.

  Return:
    loss: total loss of the output.
    outputs: a extended list with same type of content.
    labels: a extended list with same type of content.
  '''

  with torch.no_grad():
    ### MODIFY START HERE ###
    loss = criterion(output, label)
    output = np.argmax(output.cpu().numpy(), axis=-1).tolist()
    label = label.cpu().numpy().astype(np.int).tolist()
    ### END HERE ###

    ### ONLY IF YOU HAVE MULTI LABELS ###
    ### MODIFY START HERE ###
    outputs.extend(output)
    labels.extend(label)
    ### END HERE ###
  return loss.item()

def run(args):
  # Initial parameters and build directories
  settings.USE_CUDA = args.use_cuda

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  # Decide output stream
  if args.shell_print == 'file':
    settings.SHELL_OUT_FILE = open(args.output_dir + 'shell_out', 'a+',
                                   encoding='utf-8')
  else:
    settings.SHELL_OUT_FILE = sys.stdout

  # Build model and data reader/processor
  ### MODIFY START HERE ###
  ''' Shift to your own tokenizer, processor, and reader '''
  tokenizer = BertTokenizer.from_pretrained(
                  args.bert_dir + args.bert_file + '-vocab.txt')
  processor = MRPCProcessor(tokenizer, args.max_seq_length)
  reader = MRPCReader(args.batch_size)
  ### END HERE ###

  # Load/Write labels
  label_path = os.path.join(args.output_dir + 'labels.txt')
  print('Loading labels', file=settings.SHELL_OUT_FILE, flush=True)
  if os.path.exists(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
      contents = f.read()
    labels_dict = ast.literal_eval(contents)
  else:
    labels_dict = reader.get_labels()
    with open(label_path, 'w', encoding='utf-8') as f:
      pprint.pprint(labels_dict, stream=f)

  # Init
  ### MODIFY START HERE ###
  ''' Shift to your own metrics '''
  best_acc = 0
  best_recall = 0
  best_fval = 0
  best_loss = 1e9
  ### END HERE ###

  if args.do_train:
    # Create model
    model, optimizer, criterion = make_model(args)
    # Load model if it exists
    file_name = args.output_dir + 'model_' + args.suffix
    step = 0
    if os.path.exists(file_name):
      state = torch.load(file_name)
      model.load_state_dict(state['state_dict'])
      step = state['step']
    if args.multi_gpu:
      model = nn.DataParallel(model)
    model = model.cuda() if settings.USE_CUDA else model

    train_examples = reader.get_train_examples(args.data_dir)
    total_train_examples = len(train_examples)
    for ep in range(args.epoch):
      print("######## Training ########",
            file=settings.SHELL_OUT_FILE, flush=True)
      print('Epoch:', ep,
            file=settings.SHELL_OUT_FILE, flush=True)
      loss_train = []
      model.train()
      print("\rTrain Step: {} Loss: {}".format(step, 0),
            file=settings.SHELL_OUT_FILE, flush=True) # end='\r', 

      for i, example in enumerate(train_examples):
        step += 1

        ### MODIFY START HERE ###
        '''
        Adapt to output that your processor give, send them to models,
        and calculate loss
        '''
        inputs = processor.convert_examples_to_tensor(example)
        labels = processor.convert_labels_to_tensor(example.labels,
                                                    labels_dict)
        prediction = model(*inputs)
        loss = criterion(prediction, labels)
        ### END HERE ###

        loss_train.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
          print("\rTrain Step: {} Loss: {}".format(step, loss.item()),
                file=settings.SHELL_OUT_FILE, flush=True) # end='\r', 
          if args.do_eval:
            print("\n######## Evaluating ########",
                  file=settings.SHELL_OUT_FILE, flush=True)
            eval_examples = reader.get_dev_examples(args.data_dir)
            output_eval = []
            label_eval = []
            loss_eval_all = 0
            model.eval()
            total_eval_examples = len(eval_examples)
            with torch.no_grad():
              print("\rEval Step: {}/{}".format(0, total_eval_examples),
                    end='\r', file=settings.SHELL_OUT_FILE, flush=True)
              for i, example in enumerate(eval_examples):
                if (i + 1) % 100 == 0:
                  print("\rEval Step: {}/{}".format(i + 1,
                                                    total_eval_examples),
                        end='\r', file=settings.SHELL_OUT_FILE, flush=True)
    
                ### MODIFY START HERE ###
                '''
                Adapt to output that your processor give, send them to models,
                and calculate loss
                '''
                inputs = processor.convert_examples_to_tensor(example)
                labels = processor.convert_labels_to_tensor(example.labels,
                                                            labels_dict)
                prediction = model(*inputs)
                ### END HERE ###
    
                loss = evaluate(criterion, prediction, labels,
                                output_eval, label_eval)
                loss_eval_all += loss
    
              print("\rEval Step: {}/{}".format(total_eval_examples,
                                                total_eval_examples),
                    file=settings.SHELL_OUT_FILE, flush=True)
    
              loss_eval_all /= total_eval_examples
              print('Loss:', loss_eval_all,
                    file=settings.SHELL_OUT_FILE, flush=True)
    
              ### MODIFY START HERE ###
              ''' Replace by your own evaluation funciton.  '''
              acc, recall, fval, _ = \
                precision_recall_fscore_support(label_eval, output_eval,
                                                average='binary')
              print("Accuracy:", acc,
                    file=settings.SHELL_OUT_FILE, flush=True)
              print("Recall:", recall,
                    file=settings.SHELL_OUT_FILE, flush=True)
              print("F-score", fval,
                    file=settings.SHELL_OUT_FILE, flush=True)
              ### END HERE ###
    
              save_model = copy.deepcopy(model)
              save_model = save_model.module.cpu() if args.multi_gpu \
                             else save_model.cpu()

              prefix = args.output_dir + 'model_'
              # save last model
              save_best_model(model, prefix, 'last', step)
    
              ### MODIFY START HERE ###
              ''' Replace following codes by your own metrics '''
              # save model with best accuracy on dev set
              if acc > best_acc:
                best_acc = acc
                save_best_model(model, prefix, 'acc', step)
              # save model with best recall on dev set
              if recall > best_recall:
                best_recall = recall
                save_best_model(model, prefix, 'recall', step)
              # save model with best f1-score on dev set
              if fval > best_fval:
                best_fval = fval
                save_best_model(model, prefix, 'fval', step)
              # save model with best loss on dev set
              if loss_eval_all < best_loss:
                best_loss = loss_eval_all
                save_best_model(model, prefix, 'loss', step)
              ### END HERE ###
              print(file=settings.SHELL_OUT_FILE, flush=True)

    print("\rTrain Step: {} Loss: {}".format(step,
                                           sum(loss_train) / \
                                             len(train_examples)),
          file=settings.SHELL_OUT_FILE, flush=True)

  if args.do_predict:
    print("######### Testing ########",
          file=settings.SHELL_OUT_FILE, flush=True)
    test_examples = reader.get_test_examples(args.data_dir)

    ### MODIFY START HERE ###
    ''' Replace by your own metrics '''
    for suffix in ['last', 'acc', 'recall', 'fval', 'loss']:
    ### END HERE ###

      # Create model
      model, optimizer, criterion = make_model(args)
      # Load model if it exists
      file_name = args.output_dir + 'model_' + suffix

      if os.path.exists(file_name):
        state = torch.load(file_name)
        model.load_state_dict(state['state_dict'])
      else:
        continue
      if args.multi_gpu:
        model = nn.DataParallel(model)
      model = model.cuda() if settings.USE_CUDA else model

      loss_test = 0
      output_test = []
      label_test = []
      model.eval()
      total_test_examples = len(test_examples)
      with torch.no_grad():
        for i, example in enumerate(test_examples):
          if (i + 1) % 100 == 0:
            print("\rTest Step: {}/{}".format(i + 1, total_test_examples),
                  end='\r', file=settings.SHELL_OUT_FILE, flush=True)

          ### MODIFY START HERE ###
          '''
          Adapt to output that your processor give, send them to models,
          and calculate loss
          '''
          inputs = processor.convert_examples_to_tensor(example)
          labels = processor.convert_labels_to_tensor(example.labels,
                                                      labels_dict)
          predictions = model(*inputs)
          ### END HERE ###

          loss = evaluate(criterion, predictions, labels,
                          output_test, label_test)
          loss_test += loss
    
        print("\n#### " + suffix.upper() + " ####",
              file=settings.SHELL_OUT_FILE, flush=True)
        loss_test /= total_test_examples
        print("Loss:", loss_test, file=settings.SHELL_OUT_FILE, flush=True)

        output_file_name = args.output_dir + 'result_' + suffix
        with open(output_file_name, 'w', encoding='utf-8') as f:

          ### MODIFY START HERE ###
          ''' Replace by your own output format code. '''
          pprint.pprint(output_test, f)
          ### END HERE ###

        ### MODIFY START HERE ###
        ''' Replace by your own evaluation funciton.  '''
        acc, recall, fval, _ = \
          precision_recall_fscore_support(label_test, output_test,
                                          average='binary')
        print("Accuracy:", acc,
              file=settings.SHELL_OUT_FILE, flush=True)
        print("Recall:", recall,
              file=settings.SHELL_OUT_FILE, flush=True)
        print("F-score", fval,
              file=settings.SHELL_OUT_FILE, flush=True)
        ### END HERE ###
        print(file=settings.SHELL_OUT_FILE, flush=True)

if __name__ == '__main__':
  run(args)
