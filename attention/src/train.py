#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Training"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import consts as C
import utils
from seq2seq import init_model

argparser = argparse.ArgumentParser("PA4 Training Argparser")

# path flags
path_args = argparser.add_argument_group("Path Hyperparameters")
path_args.add_argument(
    '--data_dir', default='./outputs/data', help='path to data directory')
path_args.add_argument(
    '--model_dir', default='./outputs/model', help='path to model outputs')
path_args.add_argument(
    '--glove_dir', default='./glove.6B', help='path to glove dir; should be specified when using GloVe')

# model flags
model_args = argparser.add_argument_group('Model Definition Hyperparameters')
model_args.add_argument(
    '--model', type=str.lower, default='seq2seq', help='which model to init and train',
    choices=['seq2seq', 'bahdanau', 'luong_dot', 'luong_general', 'luong_concat'])
model_args.add_argument(
    '--embed_dim', type=int, default=100, help='embedding dimension')
model_args.add_argument(
    '--rnn', type=str.lower, default='lstm', choices=['gru', 'lstm'],
    help='type of rnn to use in encoder and decoder')
model_args.add_argument(
    '--num_layers', type=int, default=2, help='number of rnn layers in encoder and decoder')
model_args.add_argument(
    '--hidden_dim', type=int, default=100, help='rnn hidden dimension')
model_args.add_argument(
    '--dropout', type=float, default=0.2, help='dropout probability')
model_args.add_argument(
    '--finetune_pretrained', action='store_true', help='whether to make pretrained embeddings trainable')

# experiment flags
exp_args = argparser.add_argument_group('Experiment Hyperparameters')
exp_args.add_argument(
    '--epochs', type=int, default=1000, help='number of training iterations')
exp_args.add_argument(
    '--eval_every', type=int, default=5, help='interval of epochs to perform evaluation on dev set')
exp_args.add_argument(
    '--batch_size', type=int, default=128, help='size of mini batch')
exp_args.add_argument(
    '--sample_size', type=int, default=-1, help='sample size')
exp_args.add_argument(
    '--learning_rate', type=float, default=0.001, help='learning rate')
model_args.add_argument(
    '--teacher_forcing_ratio', type=float, default=0.5, help='teacher forcing ratio')
model_args.add_argument(
    "--bidirectional", action='store_true', help="RNN Bidirectional"
)
exp_args.add_argument(
    '--seed', type=int, default=1334, help='seed value for replicability')
exp_args.add_argument(
    '--resume_training', action='store_true', help='whether to resume training')
exp_args.add_argument(
    '--use_gpu', action='store_true', help='whether to use gpu')


def collate_fn(batch):
    """collator function used to construct padded mini-batch

    In addition to padded source and target sequences, also returns a (`batch_size`)
    array of valid lengths for both source and target. Valid lengths here refer
    to non-padded tokens. As for `tree_lengths`, it offsets by -1 to ignore <bos>.

    Args:
      batch: Tuple of source and target sequences

    Returns:
      a single batch
    """
    (sents, trees) = zip(*batch)

    sent_lengths = torch.tensor([len(x) for x in sents])
    tree_lengths = torch.tensor([len(y) - 1 for y in trees])  # -1 excludes <bos>
    sents = pad_sequence(sents, batch_first=True)  # padding_value: 0
    trees = pad_sequence(trees, batch_first=True)  # padding_value: 0

    return sents, trees, sent_lengths, tree_lengths


#################################### TRAIN #####################################
def train(args: argparse.Namespace):
    """main training function"""
    # sentinel
    assert args.glove_dir is None or os.path.exists(args.glove_dir)

    ### setup
    utils.set_seed(args.seed)
    if args.use_gpu:
        device = torch.device('cuda' if utils.use_gpu() else 'cpu')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # string formatter for model state dict filename
    model_path_template = os.path.join(args.model_dir, C.MODEL_PT_FORMATTER)

    ### load raw ptb
    ptb_train_raw = utils.load_ptb_dataset(args.data_dir, 'train')
    ptb_dev_raw = utils.load_ptb_dataset(args.data_dir, 'dev')

    sample_size = args.sample_size
    ptb_train_raw = (ptb_train_raw[0][:sample_size], ptb_train_raw[1][:sample_size])

    ### load vocab
    sent_vocab, tree_vocab = utils.load_vocabs(args.data_dir)
    sent_itos, sent_stoi = sent_vocab
    tree_itos, tree_stoi = tree_vocab
    print(f"\nVocab Info:\n Sent: {len(sent_itos)}\n Tree: {len(tree_itos)} => {', '.join(tree_itos)}")

    utils.validate_vocab(sent_stoi)
    utils.validate_vocab(tree_stoi, is_target=True)

    ### vectorized dataset
    ptb_dev = utils.PTB('dev', ptb_dev_raw, sent_vocab, tree_vocab)
    ptb_train = utils.PTB('train', ptb_train_raw, sent_vocab, tree_vocab)

    ### data loader init
    train_dataloader = DataLoader(
        ptb_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(
        ptb_dev, batch_size=args.batch_size, collate_fn=collate_fn)

    ### model init
    net = init_model(args, input_dim=len(sent_itos), output_dim=len(tree_itos),
                     tree_stoi=tree_stoi, device=device)
    num_trainable_params = utils.count_trainable_parameters(net)
    print(f" => Number of Trainable Params: {num_trainable_params}")

    # maybe resume training by loading from previous checkpoint
    model_pt_file, prev_best_bleu = utils.locate_state_dict_file(args.model_dir)
    if args.resume_training and model_pt_file:
        model_pt_path = os.path.join(args.model_dir, model_pt_file)
        print(f"Resume training with BLEU score {prev_best_bleu} from {model_pt_path}")
        net.load_state_dict(torch.load(model_pt_path))
        print(torch.load(model_pt_path).keys())
    else:
        if args.resume_training:
            print("[!] unable to find previous checkpoint, begin training from scratch")
        net.xavier_init_weights()
    net = net.to(device)

    ### loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tree_stoi[C.PAD])
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    print("\nBegin Training..")
    best_bleu = 0. if prev_best_bleu < 0 else prev_best_bleu
    for i in range(args.epochs):
        epoch_loss = 0

        num_correct = 0
        num_tokens = 0

        net.train()
        for batch in tqdm.tqdm(train_dataloader, desc=f'[Training {i + 1}/{args.epochs}]'):
            sents, trees, sent_lengths, tree_lengths = utils.to_device(batch, device)

            # same shape as sents
            attn_mask = sents == sent_stoi[C.PAD]

            # (`batch_size`, `tgt_seq_len`, `vocab_size`) where `tgt_seq_len` is the
            # number of all non-PAD tokens. Hence it includes <bos> and <eos>. Note,
            # however, that `tree_lengths` is the number of all non-PAD and non-BOS
            # tokens. So `tree_lengths` = `tgt_seq_len` - 1.
            output, _ = net(sents, trees, sent_lengths, mask=attn_mask,
                            teacher_forcing_ratio=args.teacher_forcing_ratio)

            ### loss
            # (`batch_size` * `tgt_seq_len` - 1, `vocab_size`), -1 excludes last timestep prediction
            output_ = output[:, :-1].contiguous().view(-1, output.size(-1))
            # (`batch_size` * `tgt_seq_len` - 1), -1 excludes <bos>
            trees_ = trees[:, 1:].contiguous().view(-1)  # exclude bos
            loss = criterion(output_, trees_)

            ### token-level accuracy
            preds = output_.argmax(-1)  # (`batch_size` * `tgt_seq_len` - 1)
            num_correct += (preds == trees_).masked_fill_(trees_ == 0, False).sum()
            num_tokens += tree_lengths.sum()  # includes <eos>

            ### optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()

        acc = (num_correct.item() / num_tokens.item()) * 100
        print(' Training Loss: {:.2f} Token-level Accuracy: {:.2f}%'.format(epoch_loss, acc))

        ### eval
        if (i + 1) % args.eval_every == 0:
            dev_loss = 0
            dev_num_correct = 0
            dev_num_tokens = 0
            dev_preds = []

            net.eval()
            for batch in tqdm.tqdm(dev_dataloader, desc=f'[Dev Eval]'):
                sents, trees, sent_lengths, tree_lengths = utils.to_device(batch, device)

                # same shape as sents
                attn_mask = sents == sent_stoi[C.PAD]

                # (`batch_size`, `tgt_seq_len`, `vocab_size`)
                output, _ = net(sents, trees, sent_lengths, mask=attn_mask,
                                teacher_forcing_ratio=0.)

                # (`batch_size` * `tgt_seq_len`-1, `vocab_size`)
                output_ = output[:, :-1].contiguous().view(-1, output.shape[-1])
                # (`batch_size` * `tgt_seq_len`-1)
                trees_ = trees[:, 1:].contiguous().view(-1)  # exclude bos
                loss = criterion(output_, trees_)
                dev_loss += loss.item()

                ### token-level accuracy
                preds = output_.argmax(-1)
                dev_num_correct += (preds == trees_).masked_fill_(trees_ == 0, False).sum()
                dev_num_tokens += tree_lengths.sum()

                # (`batch_size`, `tgt_seq_len`)
                output_argmax = output.argmax(-1)
                for output_argmax_, length in zip(output_argmax.tolist(), tree_lengths.tolist()):
                    output_pred = output_argmax_[:length - 1]  # -1 to discard <eos>
                    dev_preds.append(output_pred)

            dev_acc = (dev_num_correct.item() / dev_num_tokens.item()) * 100

            dev_preds_str = [utils.convert_seq(dev_pred, tree_itos, return_str=True)
                             for dev_pred in dev_preds]
            bleu = utils.raw_corpus_bleu(dev_preds_str, ptb_dev_raw[1])

            msg = ' Dev Loss: {:.3f} | Dev Token-level Accuracy: {:.2f}%'.format(dev_loss, dev_acc)
            msg += ' | Dev BLEU: {:.2f} (Current Best BLEU: {:.2f})'.format(bleu, best_bleu)
            print(msg)

            if bleu > best_bleu:
                print("[UPDATE] Best BLEU score ({:.2f} > {:.2f}) => Saving model params..".format(bleu, best_bleu))
                dev_preds_path = os.path.join(args.model_dir, C.TREE_PRED_FORMATTER.format('dev'))
                print("Exporting dev predictions at", dev_preds_path)
                utils.export_txt(dev_preds_str, dev_preds_path)
                try:  # attempt to delete previous checkpoint
                    os.remove(model_path_template.format(best_bleu))
                except FileNotFoundError:
                    pass
                best_bleu = bleu

                # export model state dict
                torch.save(net.state_dict(), model_path_template.format(best_bleu))

            ### sample prediction from last dev batch
            sample_idx = random.randint(0, len(sents) - 1)
            sample_sent = sents[sample_idx][:sent_lengths[sample_idx].item()].tolist()  # includes <eos>
            sample_sent_str = utils.convert_seq(sample_sent, sent_itos, return_str=True)
            msg = f"\nSample Dev Prediction:\n [SENT] {sample_sent_str}"

            sample_tree_len = tree_lengths[sample_idx].item()  # exclude <bos> but includes <eos>
            sample_tree = trees[sample_idx][1:sample_tree_len + 1].tolist()  # offset by 1
            sample_tree = utils.convert_seq(sample_tree, tree_itos)
            sample_tree_str = " ".join(sample_tree)
            msg += f"\n [GOLD] {sample_tree_str}"

            sample_pred = output[sample_idx].argmax(-1).tolist()[:sample_tree_len]  # includes <eos>
            sample_pred = utils.convert_seq(sample_pred, tree_itos)
            sample_pred_str = " ".join(sample_pred)
            msg += f"\n [PRED] {sample_pred_str}"

            sample_bleu = utils.raw_corpus_bleu([sample_pred_str], [sample_tree_str])
            sample_correct = sum(p == t for p, t in zip(sample_pred, sample_tree))
            sample_acc = sample_correct / len(sample_pred)
            msg += "\n [SCORE] BLEU {:.2f} | ACC: {:.2f} | LENGTH {} toks".format(sample_bleu, sample_acc,
                                                                                  len(sample_pred))

            print(msg)


def main():
    """Training script entry point"""
    begin = time.time()

    args, _ = argparser.parse_known_args()
    assert os.path.exists(args.data_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    args_path = os.path.join(args.model_dir, C.ARGS_JSON_FILENAME)

    # when args json file found from previous execution, consider loading them
    if os.path.exists(args_path):
        print("Loading args from", args_path)
        prev_args_dict = utils.load_json(args_path)
        is_different = utils.maybe_update_args(args, prev_args_dict, ignore_flags=C.TRAIN_IGNORE_FLAGS)

        if args.resume_training and is_different:
            raise ValueError("`resume_training` specified, but either `data_dir` or model definition has changed.")

    utils.export_json(vars(args), args_path)
    utils.display_args(args)

    train(args)

    utils.display_exec_time(begin, "Training")


if __name__ == '__main__':
    main()
