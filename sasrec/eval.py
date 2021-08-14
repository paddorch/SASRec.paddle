import sys
import random
import copy
import os
import argparse
import datetime
import time
import sys
import errno
import pickle
from tqdm import tqdm

import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F
import numpy as np

from model import SASRec
from utils import data_partition


def evaluate(dataset, model, epoch_train, batch_train, args, is_val=True):
    model.eval()
    T = 0
    lst_time = 0
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    before = train
    now = valid if is_val else test

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(before[u]) < 1 or len(now[u]) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if is_val:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(before[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(before[u])
        rated.add(0)
        item_idx = [now[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        t0 = time.time()
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        T += (time.time() - t0)
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
            # print(f'predict: {T}')
            # T = 0
            # print(f'log_feats: {model.T}')
            # model.T = 0
            print(f'total: {time.time() - lst_time}')
            lst_time = time.time()

    NDCG /= valid_user
    HT /= valid_user

    model.train()
    print('\nEvaluation - NDCG: {:.4f}  HIT@10: {:.4f}'.format(NDCG, HT))
    print('\n')
    if args.log_result and is_val:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.4f},{:.4f}'.format(epoch_train,
                                                       batch_train,
                                                       NDCG,
                                                       HT))

    return (NDCG, HT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SASRec training')
    # data
    parser.add_argument('--dataset_path', metavar='DIR',
                        help='path to training data csv [default: ../data/preprocessed/train_data.pkl]',
                        default='../data/preprocessed/ml-1m.txt')
    # learning
    learn = parser.add_argument_group('Learning options')
    learn.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.01]')
    learn.add_argument('--epochs', type=int, default=601, help='number of epochs for train [default: 200]')
    learn.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 50]')
    learn.add_argument('--optimizer', default='Adam',
                       help='Type of optimizer. Adagrad|Adam|AdamW are supported [default: Adagrad]')
    # model
    model_cfg = parser.add_argument_group('Model options')
    model_cfg.add_argument('--hidden_units', type=int, default=50,
                           help='hidden size of LSTM [default: 300]')
    model_cfg.add_argument('--maxlen', type=int, default=200,
                           help='hidden size of LSTM [default: 300]')
    model_cfg.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    model_cfg.add_argument('--l2_emb', type=float, default=0.0, help='penalty term coefficient [default: 0.1]')
    model_cfg.add_argument('--num_blocks', type=int, default=2,
                           help='d_a size [default: 150]')
    model_cfg.add_argument('--num_heads', type=int, default=1,
                           help='row size of sentence embedding [default: 30]')
    # device
    device = parser.add_argument_group('Device options')
    device.add_argument('--num_workers', default=0, type=int, help='Number of workers used in data-loading')
    device.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('--device', type=str, default='gpu:0')
    # experiment options
    experiment = parser.add_argument_group('Experiment options')
    experiment.add_argument('--continue_from', default='', help='Continue from checkpoint model')
    experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                            help='Enables checkpoint saving of model')
    experiment.add_argument('--checkpoint_per_batch', default=10000, type=int,
                            help='Save checkpoint per batch. 0 means never save [default: 10000]')
    experiment.add_argument('--save_folder', default='../output/',  # TODO
                            help='Location to save epoch models, training configurations and results.')
    experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
    experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
    experiment.add_argument('--log_interval', type=int, default=20,
                            help='how many steps to wait before logging training status [default: 1]')
    experiment.add_argument('--val_interval', type=int, default=100,
                            help='how many steps to wait before vaidation [default: 400]')
    experiment.add_argument('--save_interval', type=int, default=100,
                            help='how many epochs to wait before saving [default:1]')
    args = parser.parse_args()
    # paddle.set_device(args.device)

    dataset = data_partition(args.dataset_path)

    [user_train, _, _, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('\nAverage sequence length: %.2f' % (cc / len(user_train)))

    # TODO
    model = SASRec(usernum, itemnum, args)
    evaluate(dataset, model, 0, 0, args, is_val=False)
