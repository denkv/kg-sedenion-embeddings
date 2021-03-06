#!/usr/bin/env python3
import argparse
import os

import config
from models import SedeniE


def train(input, output, train, bern, dimension, lmbda, lmbda2, ent_neg_rate):
    checkpoint_dir = os.path.join(output, 'checkpoint')
    os.makedirs(checkpoint_dir)
    result_dir = os.path.join(output, 'result')
    os.makedirs(result_dir)

    con = config.Config()
    con.set_in_path(input)
    con.set_work_threads(8)
    con.set_train_times(train)
    con.set_nbatches(10)
    con.set_alpha(0.1)
    con.set_bern(bern)  # 0 or 1
    con.set_dimension(dimension)  # embedding dimensionality
    con.set_lmbda(lmbda)  # regularization in loss function
    con.set_lmbda_two(lmbda2)
    con.set_margin(1.0)
    con.set_ent_neg_rate(ent_neg_rate)
    con.set_rel_neg_rate(0)
    con.set_opt_method("adagrad")
    con.set_save_steps(1000)
    con.set_valid_steps(1000)
    con.set_early_stopping_patience(10)
    con.set_checkpoint_dir(checkpoint_dir)
    con.set_result_dir(result_dir)
    con.set_test_link(True)
    con.set_test_triple(True)
    con.init()
    con.set_train_model(SedeniE)
    con.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train', required=True, type=int)
    parser.add_argument('--bern', required=True, type=int, choices=[0, 1])
    parser.add_argument('--dimension', required=True, type=int)
    parser.add_argument('--lmbda', required=True, type=float)
    parser.add_argument('--lmbda2', required=True, type=float)
    parser.add_argument('--ent-neg-rate', required=True, type=int)
    args = parser.parse_args()
    print(f'Arguments: {vars(args)}')
    train(**vars(args))
