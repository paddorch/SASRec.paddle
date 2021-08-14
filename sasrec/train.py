#!/usr/bin/env python3
import argparse
import datetime
import errno
import sys
import random
import copy
import pickle
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import optimizer
from paddle.io import DataLoader
import numpy as np

from model import SASRec, MyBCEWithLogitLoss
from data import WarpSampler, sample_function
from utils import set_seed, data_partition
from eval import evaluate

set_seed(42)

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
device.add_argument('--gpu', type=int, default=None)
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


def train(sampler, model, args, num_batch, dataset):
    # clip gradient
    # if args.grad_clip:
    #     clip = paddle.nn.ClipGradByValue(max=args.grad_clip)
    # else:
    #     clip = None
    clip = None
    # optimization scheme
    if args.optimizer == 'Adam':
        optim = optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'Adagrad':
        optim = optimizer.Adagrad(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'AdamW':
        optim = optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)

    # loss
    # criterion = nn.BCEWithLogitsLoss()
    criterion = MyBCEWithLogitLoss()

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = paddle.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        best_pair = checkpoint.get('best_pair', None)
        model.set_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 1
        best_pair = None

    model.train()

    tot_batch = 0

    for epoch in range(start_epoch, args.epochs + 1):
        _i_batch = 0
        for i_batch in range(num_batch):
            tot_batch += 1
            # _i_batch = i_batch
            # premises = paddle.to_tensor(batch["premise"])
            # premises_lengths = paddle.to_tensor(batch["premise_length"])
            # hypothesises = paddle.to_tensor(batch["hypothesis"])
            # hypothesis_lengths = paddle.to_tensor(batch["hypothesis_length"])
            # target = paddle.to_tensor(batch["label"])
            #
            # logit, penalty = model(premises, premises_lengths, hypothesises, hypothesis_lengths)
            # loss = criterion(logit, target)
            # if args.use_penalty:
            #     loss += args.penalty_c*penalty
            # loss.backward()
            # optim.step()
            # optim.clear_grad()
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = paddle.to_tensor(u, dtype='int64'), paddle.to_tensor(seq, dtype='int64'), paddle.to_tensor(pos), paddle.to_tensor(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)    # ()
            # pos_labels, neg_labels = np.ones(pos_logits.shape), np.zeros(neg_logits.shape)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            # indices = np.where(pos != 0)
            # pos_logits, neg_logits = paddle.to_tensor(pos_logits[indices]), paddle.to_tensor(neg_logits[indices])
            # pos_labels, neg_labels = paddle.to_tensor(pos_labels[indices], dtype='float32'), paddle.to_tensor(neg_labels[indices], dtype='float32')
            # loss = criterion(pos_logits, pos_labels)
            # loss += criterion(neg_logits, neg_labels)

            targets = (pos != 0).astype(dtype='int32')
            # targets = targets.reshape((args.batch_size*args.maxlen, -1))
            loss = criterion(pos_logits, neg_logits, targets)
            # TODO
            # for param in model.item_emb.parameters():
            #     loss += args.l2_emb * paddle.norm(param)
            loss.backward()
            optim.step()
            optim.clear_grad()

            if i_batch % args.log_interval == 0:
                # corrects = paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
                # accuracy = 100.0 * corrects / args.batch_size
                print('Epoch[{}] Batch[{}] - loss: {:.8f}  lr: {:.5f}'.format(epoch,
                                                                              i_batch,
                                                                              loss.numpy()[0],
                                                                              optim._learning_rate,
                                                                              ))
            # validation
            if tot_batch % args.val_interval == 0 and i_batch != 0:
                valid_pair = evaluate(dataset, model, epoch, i_batch, args, is_val=True)
                if best_pair is None or valid_pair > best_pair:
                    file_path = '%s/SASRec_best.pth.tar' % (args.save_folder)
                    print("\r=> found better validated model, saving to %s" % file_path)
                    save_checkpoint(model,
                                    {'epoch': epoch,
                                     'optimizer': optim.state_dict(),
                                     'best_pair': best_pair},
                                    file_path)
                    best_pair = valid_pair

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/SASRec_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optim.state_dict(),
                                    'best_pair': best_pair},
                            file_path)
        print('\n')
    sampler.close()


def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    paddle.save(state, filename)


def main():
    print(paddle.__version__)
    # parse arguments
    args = parser.parse_args()
    # gpu
    if args.cuda and args.gpu:
        paddle.set_device(f"gpu:{args.gpu}")

    dataset = data_partition(args.dataset_path)

    [user_train, _, _, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('\nAverage sequence length: %.2f' % (cc / len(user_train)))

    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=args.num_workers)

    # make save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))

    # model
    model = SASRec(usernum, itemnum, args)
    print(model)
    # TODO init

    # train
    train(sampler, model, args, num_batch, dataset)


if __name__ == '__main__':
    main()
