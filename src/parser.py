import argparse
from settings import *

parser = argparse.ArgumentParser(description='Hyperparameters')
### MODIFY START HERE ###
''' Add or delete parameters that you need '''
parser.add_argument('--learning_rate', type=float, dest='lr', default=LR)
parser.add_argument('--dropout', type=float, default=DROPOUT)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--data_dir')
parser.add_argument('--output_dir', default='results/')
parser.add_argument('--shell_print', default='file', choices=['file', 'shell'],
                    help='Indicate where shell output should be redirect')
parser.add_argument('--suffix', default='last',
                    choices=['last', 'acc', 'recall', 'fval', 'loss'],
                    help='File suffix after whole name made by parameters')
parser.add_argument('--bert_dir')
parser.add_argument('--bert_file',
                    help='Bert tar.gz file name without suffix')
parser.add_argument('--warmup_propotion', type=float, default=0.1,
                    help="Proportion of training to perform linear learning "
                         "rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--l2', type=float, default=L2_NORM)
parser.add_argument('--max_seq_length', type=int, default=MAX_SEQ_LEN,
                    help='Max sequence length input to BERT')
parser.add_argument('--multi_gpu', action='store_true',
                    help='Use multi gpu')
### END HERE ###

args = parser.parse_args()
