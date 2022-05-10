import argparse
import itertools
import pickle as pkl

import numpy as np
import torch
from reservoir_ca.supervised_wade_exps.dataset import (
    get_tokenized_dataset,
    get_tokenizer,
    get_train_test_sets,
)
from reservoir_ca.supervised_wade_exps.model import Recurrent, TransformerModel
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.transformer import Transformer
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--embed-size", type=int, default=128)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--n-layers", type=int, default=1)
parser.add_argument("--subset", type=float, default=1.0)
parser.add_argument(
    "--model", type=str, default="RNN", choices=["RNN", "LSTM", "GRU", "Transformer"]
)
parser.add_argument("--output-file", type=str, default="output.pkl")


def model_needs_len(model: str) -> bool:
    if model in ["RNN", "LSTM", "GRU"]:
        return True
    else:
        return False


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    tokenizer = get_tokenizer()
    subset = args.subset
    # Train is an iterator returning tuples (label, sentence)
    train_iter, test_iter = get_train_test_sets()
    tokenizer.train_from_iterator(
        map(lambda x: x[1], itertools.chain(train_iter, test_iter))
    )

    if args.model in ["RNN", "LSTM", "GRU"]:
        model = Recurrent(
            tokenizer.get_vocab_size(),
            embed_size=args.embed_size,
            num_output=2,
            rnn_model=args.model,
            hidden_size=args.hidden_size,
            num_layers=args.n_layers,
        )
    elif args.model == "Transformer":
        model = TransformerModel(
            tokenizer.get_vocab_size(),
            args.embed_size,
            8,
            args.hidden_size,
            args.n_layers,
            2,
        )
    else:
        raise ValueError("Unknown model type")

    print("Parameters", sum(p.reshape(-1).size()[0] for p in model.parameters()))
    model.to(device)
    loss = CrossEntropyLoss()
    opt = optim.Adam(model.parameters())

    eval_batch_size = 32

    tokenized_test = get_tokenized_dataset(test_iter, tokenizer)
    test_inputs = [torch.Tensor(i[0]).long().to(device) for i in tokenized_test]
    test_labels = torch.Tensor([i[1] for i in tokenized_test]).long().to(device)

    tokenized_train = get_tokenized_dataset(train_iter, tokenizer)
    inputs = [torch.Tensor(i[0]).long().to(device) for i in tokenized_train]
    labels = torch.Tensor([i[1] for i in tokenized_train]).long().to(device)
    n_inputs = len(inputs)

    if subset < 1.0:
        subset_idx = np.random.choice(
            range(len(test_inputs)), size=int(subset * len(test_inputs))
        )
        test_inputs = [test_inputs[i] for i in subset_idx]
        test_labels = test_labels[subset_idx]

        inputs = [inputs[i] for i in subset_idx]
        labels = labels[subset_idx]
        n_inputs = len(inputs)

    if not model_needs_len(args.model) and torch.cuda.is_available():
        rand_batch = torch.randint(
            0, 2, size=(max(i.shape[0] for i in inputs), batch_size), device=device
        ).long()
        rand_labels = torch.randint(0, 2, size=(batch_size,), device=device).long()

        output = model(rand_batch)
        error = loss(output, rand_labels)
        error.backward()
        rand_batch = torch.randint(
            0,
            2,
            size=(max(i.shape[0] for i in test_inputs), eval_batch_size),
            device=device,
        ).long()
        output = model(rand_batch)

    #     inputs = pad_sequence(inputs)
    #     test_inputs = pad_sequence(test_inputs)
    #     n_inputs = inputs.shape[1]

    n_steps = 0
    all_accuracies = []
    for epoch in range(args.epochs):

        indices = np.random.permutation(range(n_inputs))

        for b in range(0, len(indices), batch_size):
            model.train()
            for param in model.parameters():
                param.grad = None
            batch_labels = labels[indices[b : b + batch_size]]
            # if model_needs_len(args.model):

            batch_input = [inputs[i] for i in indices[b : b + batch_size]]
            batch_lengths = (
                torch.Tensor([i.size() for i in batch_input]).reshape(-1).long()
            )
            padded_input = pad_sequence(batch_input)
            if model_needs_len(args.model):
                output = model(padded_input, batch_lengths)
            else:
                output = model(padded_input)
            #     padded_input = inputs[:, indices[b : b + batch_size]]

            error = loss(output, batch_labels)
            error.backward()
            opt.step()

            n_steps += padded_input.shape[1]

            if b % (50 * batch_size) == 0:
                model.eval()
                with torch.no_grad():
                    val_error = 0
                    val_accuracy = 0
                    for s in range(0, n_inputs, eval_batch_size):
                        batch_labels = test_labels[s : s + eval_batch_size]
                        batch_input = test_inputs[s : s + eval_batch_size]
                        batch_lengths = (
                            torch.Tensor([i.size() for i in batch_input])
                            .reshape(-1)
                            .long()
                        )
                        padded_input = pad_sequence(batch_input)
                        if model_needs_len(args.model):
                            output = model(padded_input, batch_lengths)
                        else:
                            output = model(padded_input)
                        error = loss(output, batch_labels)

                        val_error += error.item() * padded_input.shape[1]
                        val_accuracy += (
                            output.argmax(1) == batch_labels
                        ).float().mean().item() * padded_input.shape[1]

                    val_error /= len(test_inputs)
                    val_accuracy /= len(test_inputs)
                    print("Steps:", n_steps, "Accuracy:", val_accuracy)
                    all_accuracies.append((n_steps, val_accuracy))

    pkl.dump((all_accuracies, vars(args)), open(args.output_file, "wb"))
