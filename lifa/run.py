# arguments
# --pretrained_model    ["bert", "phobert"]
# --dataset             ["aivivn", "tiki"]

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)

parser.add_argument('--pretrained_model', type=str)
parser.add_argument('--dataset', type=str)

parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--nrows', default=None, type=int)

parser.add_argument('--model_name', default='Moe_Gating', type=str) #Moe_Gating, Moe_Concate, Moe_GatingGLU

args = parser.parse_args()



if args.pretrained_model == "bert":
    from main.main import *
    train_bert(args.dataset, args.num_epochs, args.batch_size, args.learning_rate)

elif args.pretrained_model == "phobert":
    from main.main import *
    train_phobert(args.dataset, args.num_epochs, args.batch_size, args.learning_rate)

elif args.pretrained_model == "xlm":
    from main.main import *
    train_xlm(args.dataset, args.num_epochs, args.batch_size, args.learning_rate)

elif args.pretrained_model == "lstmcnn":
    from main.main import *
    train_lstmcnn(args.dataset, args.num_epochs, args.batch_size, args.learning_rate)

elif args.pretrained_model == "moe":
    from main.main import *
    #train_moe(args.dataset, args.num_epochs, args.batch_size, args.learning_rate)
    train_moe_phobert_bert_lstmcnn(args.dataset, args.num_epochs, args.batch_size, args.learning_rate, args.model_name)
