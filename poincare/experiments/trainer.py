
import os
import sys
import logging
import argparse
from datetime import datetime
import time
from argparse import ArgumentParser
sys.path.insert(0, os.getcwd())

import pickle
import numpy as np
import pandas as pd
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors, PoincareRelations

from utils import data_loader


def train_run(args):
    # create experiment name from args
    # create log folder from exp name
    # save model from exp name
    
    exp_name = 'HB'+'time' + str(datetime.now()) + '_EXP' + str(args.train_dir) + \
    '_prbt' + str(args.prob_threshold) + '_reg' + str(args.reg_coef) + \
    '_dim' + str(args.embed_dim) + '_lr' + str(args.learning_rate) + \
    '_epoc' + str(args.epochs) + '_burnin' + str(args.burn_in)
    
   
    logging.basicConfig(level=logging.INFO)
    for arg in vars(args):
        print arg, getattr(args, arg)
    return

def main():
    # Basics
    parser = argparse.ArgumentParser(description='Process the model parameters')
    parser.add_argument('--random_seed', type=int, default=20180112, help='random seed for model')
    parser.add_argument('--save', type=bool, default=True, help='Save the model')
    parser.add_argument('--log_folder', type=str, default='./log/', help='Log folder location')
    parser.add_argument('--params_folder', type=str, default='./params/', help='Params folder location')
    
    # Dataset 
    parser.add_argument('--train_dir', type=str, default='./data/book_data/book_small', help='Directory for data files')
    
    parser.add_argument('--trn_file', type=str, default='book_train.txt', help='Training file')
    parser.add_argument('--trn_eval', type=str, default='book_train_eval.txt', help='Trn eval file')
    parser.add_argument('--dev_file', type=str, default='book_dev.txt', help='Dev data file')
    parser.add_argument('--tst_file', type=str, default='book_test.txt', help='Test data file')
    parser.add_argument('--marg_prob_file', type=str, default='book_marginal_prob.txt', help='which marginal probability file to use')

    parser.add_argument('--prob_threshold', type=float, default=0.5, help='Conditional probability threshold')
    
    # Model defn
    parser.add_argument('--embed_dim', type=int, default=50, help='Number of dimensions of the trained model')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--negative', type=int, default=10, help='Number of negative samples to use')
    parser.add_argument('--reg_coef', type=float, default=1.0, help='Coefficient used for l2-regularization while training')
    parser.add_argument('--burn_in', type=int, default=10, help='Number of epochs to use for burn-in initialization')
    parser.add_argument('--burn_in_alpha', type=float, default=0.01, help='Learning rate for burn-in initialization')
    parser.add_argument('--init_range', type=tuple, default=(-0.001, 0.001), help='2-tuple Range within which the vectors are randomly initialized')
    
    # Model trainig
    parser.add_argument('--epochs', type=int, default=1, help='Number of iterations (epochs) over the corpus')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of iterations (epochs) over the corpus')
    parser.add_argument('--print_every', type=int, default=100, help='Prints progress and average loss after every `print_every` batches')
    
    # Parse the arguments                              
    args = parser.parse_args()
    
    # Run the trianing function with the args
    train_run(args)
    return

if __name__ == '__main__':
    main()
