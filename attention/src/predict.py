#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Inference"""
import argparse
import os
import time

import torch
import tqdm

import consts as C
import utils
from seq2seq import init_model

argparser = argparse.ArgumentParser("PA4 Inference Argparser")
argparser.add_argument('--model_dir', required=True, help='path to model dir')


def predict(args: argparse.Namespace):
    """main inference function"""
    # sentinel
    assert os.path.exists(args.model_dir), f"`model_dir` at {args.model_dir} doesn't exist"

    ### setup
    utils.set_seed(args.seed)
    device = torch.device('cuda' if utils.use_gpu() else 'cpu')
    #device = torch.device('cpu')
    print("Using device:", device)

    ### load raw test ptb
    ptb_test_raw = utils.load_ptb_dataset(args.data_dir, 'test')

    ### load vocab
    sent_vocab, tree_vocab = utils.load_vocabs(args.data_dir)
    sent_itos, sent_stoi = sent_vocab
    tree_itos, tree_stoi = tree_vocab
    utils.validate_vocab(sent_stoi)
    utils.validate_vocab(tree_stoi, is_target=True)

    ### vectorized PTB test
    ptb_test = utils.PTB('test', ptb_test_raw, sent_vocab, tree_vocab)

    ### model init
    net = init_model(
        args, input_dim=len(sent_itos), output_dim=len(tree_itos),
        tree_stoi=tree_stoi, device=device
    )
    num_trainable_params = utils.count_trainable_parameters(net)
    print(f" => Number of Trainable Params: {num_trainable_params}")

    ### load state dict
    model_pt_file, prev_best_bleu = utils.locate_state_dict_file(args.model_dir)
    model_pt_path = os.path.join(args.model_dir, model_pt_file)
    net.load_state_dict(torch.load(model_pt_path))
    print(f"Successfully loaded checkpoint from {model_pt_path}")
    net = net.to(device)

    print("\nBegin Inference..")

    num_correct = 0
    num_tokens = 0

    ### prediction
    preds = []
    for i in tqdm.trange(len(ptb_test), desc=f'[Inference]'):
        sent, tree = utils.to_device(ptb_test[i], device)
        sent_length = torch.tensor([len(sent)], dtype=torch.long, device=device)

        # same shape as sents
        attn_mask = (sent == sent_stoi[C.PAD]).unsqueeze(0)

        ### arbitrary threshold to prefer for-loop, preventing possible infinite loop
        max_pred_tree_len = len(sent) * 3

        ### Encoder step
        enc_outputs, state = net.encoder(sent.unsqueeze(0), sent_length)

        ### Unrolling Decoder
        input_ = torch.tensor([[tree_stoi[C.BOS]]], dtype=torch.long, device=device)

        pred = []
        for j in range(max_pred_tree_len):
            if net.is_attentional:
                output, state = net.decoder(input_, state, enc_outputs, mask=attn_mask)
            else:
                output, state = net.decoder(input_, state)
            input_ = output.argmax(-1)

            try:
                # +1 offset for bos at the beginning of tree
                if (tree[j + 1] == input_.view(1)).item():
                    num_correct += 1
            except IndexError:
                pass
            num_tokens += 1

            input_ = input_.view(1, 1)

            tok_str = tree_itos[input_.item()]
            if tok_str == C.EOS:
                break
            pred.append(tok_str)  # pred never contains EOS
        preds.append(" ".join(pred))

    acc = (num_correct / num_tokens) * 100

    # export
    preds_path = os.path.join(args.model_dir, C.TREE_PRED_FORMATTER.format('test'))
    print("Exporting test predictions at", preds_path)
    utils.export_txt(preds, preds_path)

    # bleu
    bleu = utils.raw_corpus_bleu(preds, ptb_test_raw[1])
    print("\nTest BLEU: {:.2f} | Test Token-level Accuracy: {:.2f}".format(bleu, acc))


def main():
    """inference script entry point"""
    begin = time.time()

    # parse args
    pred_args, _ = argparser.parse_known_args()
    assert os.path.exists(pred_args.model_dir)

    args_path = os.path.join(pred_args.model_dir, C.ARGS_JSON_FILENAME)
    prev_args_dict = utils.load_json(args_path)

    args = argparse.Namespace()
    args.__dict__.update(prev_args_dict)
    assert os.path.abspath(args.model_dir) == os.path.abspath(pred_args.model_dir), \
        f"{args.model_dir} vs {pred_args.model_dir}"

    utils.display_args(args)

    predict(args)

    utils.display_exec_time(begin, "Inference")


if __name__ == '__main__':
    main()
