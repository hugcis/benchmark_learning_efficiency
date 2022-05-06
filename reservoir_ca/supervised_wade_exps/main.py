import argparse
import itertools

import numpy as np
import torch
from torch import optim
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss
from torch.nn.modules.transformer import Transformer
from reservoir_ca.supervised_wade_exps.dataset import (
    get_tokenized_dataset,
    get_tokenizer,
    get_train_test_sets,
)
from reservoir_ca.supervised_wade_exps.model import Recurrent
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--embed-size", type=int, default=100)
parser.add_argument("--hidden-size", type=int, default=100)
parser.add_argument("--subset", type=float, default=1.0)
parser.add_argument(
    "--model", type=str, default="RNN", choices=["RNN", "LSTM", "GRU", "Transformer"]
)

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
            rnn_model="LSTM",
            hidden_size=args.hidden_size,
        )
    else:
        model = Transformer()
    loss = CrossEntropyLoss()
    opt = optim.Adam(model.parameters())

    eval_batch_size = 128
    tokenized_test = get_tokenized_dataset(test_iter, tokenizer)
    test_inputs = [torch.Tensor(i[0]).long() for i in tokenized_test]
    test_labels = torch.Tensor([i[1] for i in tokenized_test]).long()
    tokenized_train = get_tokenized_dataset(train_iter, tokenizer)
    inputs = [torch.Tensor(i[0]).long() for i in tokenized_train]
    labels = torch.Tensor([i[1] for i in tokenized_train]).long()
    if subset < 1.0:
        subset_idx = np.random.choice(
            range(len(test_inputs)), size=int(subset * len(test_inputs))
        )
        test_inputs = [test_inputs[i] for i in subset_idx]
        test_labels = test_labels[subset_idx]

        inputs = [inputs[i] for i in subset_idx]
        labels = labels[subset_idx]

    n_steps = 0
    all_accuracies = []
    for epoch in range(args.epochs):
        indices = np.random.permutation(range(len(inputs)))

        for b in range(0, len(indices), batch_size):
            opt.zero_grad()
            batch_input = [inputs[i] for i in indices[b : b + batch_size]]
            batch_lengths = (
                torch.Tensor([i.size() for i in batch_input]).reshape(-1).long()
            )
            padded_input = pad_sequence(batch_input)
            batch_labels = labels[indices[b : b + batch_size]]

            output = model(padded_input, batch_lengths)
            error = loss(output, batch_labels)
            error.backward()
            opt.step()

            n_steps += len(batch_input)

            if b % (10 * batch_size) == 0:
                with torch.no_grad():
                    val_error = 0
                    val_accuracy = 0
                    for b in range(0, len(test_inputs), eval_batch_size):
                        batch_input = test_inputs[b : b + eval_batch_size]
                        batch_lengths = (
                            torch.Tensor([i.size() for i in batch_input])
                            .reshape(-1)
                            .long()
                        )
                        padded_input = pad_sequence(batch_input)
                        batch_labels = labels[b : b + eval_batch_size]

                        output = model(padded_input, batch_lengths)
                        error = loss(output, batch_labels)

                        val_error += error.item() * len(batch_input)
                        val_accuracy += (
                            output.argmax(1) == batch_labels
                        ).float().mean().item() * len(batch_input)

                    val_error /= len(test_inputs)
                    val_accuracy /= len(test_inputs)
                    print(n_steps, val_accuracy)
                    all_accuracies.append((n_steps, val_accuracy))
