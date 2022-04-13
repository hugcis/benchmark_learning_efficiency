"""The decoders for reading and predicting outputs from the reservoirs."""
import logging
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from torch import nn, optim
from torch.nn import functional

__all__ = [
    "LogisticRegression",
    "SGDClassifier",
    "LinearSVC",
    "SVC",
    "StandardScaler",
    "MLPClassifier",
    "RandomForestClassifier",
]


class ExperimentData(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.X = x
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class ConvNetwork(nn.Module):
    """A Convolutional neural network for the ConvClassifier."""

    def __init__(
        self,
        channels: Sequence[int],
        inp_dim: int,
        out_dim: int,
        kernel_sizes: Optional[Sequence[int]] = None,
        avg_pool: bool = True,
    ) -> None:
        super().__init__()

        self.avg_pool = avg_pool
        if len(channels) < 1:
            raise ValueError("Argument channels must have more than one value")
        if kernel_sizes is None:
            kernel_sizes = [3] * (len(channels) + 1)
        elif len(kernel_sizes) < len(channels) + 1:
            raise ValueError("There must be len(channels) + 1 specified kernel sizes")

        convs = []
        self.channels = channels
        self.out_dim = out_dim
        self.conv_first = nn.Conv1d(inp_dim, channels[0], kernel_sizes[0])
        for i in range(len(channels) - 1):
            convs.append(nn.Conv1d(channels[i], channels[i + 1], kernel_sizes[i + 1]))
        self.convs = nn.ModuleList(convs)
        self.conv_last = nn.Conv1d(channels[-1], out_dim, kernel_sizes[-1])

    def forward(self, X):
        X = torch.relu(self.conv_first(X))
        for conv in self.convs:
            X = torch.relu(conv(X))
        X = self.conv_last(X)
        if self.avg_pool:
            return functional.avg_pool1d(X, kernel_size=X.shape[-1])
        else:
            return functional.max_pool1d(X, kernel_size=X.shape[-1])


class OptType(Enum):
    LBGFS = 1
    SGD = 2
    ADAM = 3


class ConvClassifier(BaseEstimator, ClassifierMixin):
    """The convolutional neural network classifier."""

    conv_network: Optional[ConvNetwork]

    def __init__(
        self,
        channels: Sequence[int],
        verbose: bool = False,
        opt_type: OptType = OptType.ADAM,
    ):
        self.conv_network = None
        self.channels = list(channels)
        self.verbose = verbose
        self.opt_type = opt_type

    def fit(
        self, X, y, batch_size: Optional[int] = None, epochs: int = 10
    ) -> "ConvClassifier":
        # Check classes
        self.classes_ = unique_labels(y)
        self.inverse_classes_ = np.zeros(self.classes_.max() + 1, dtype=int)
        for i, c in enumerate(self.classes_):
            self.inverse_classes_[c] = i

        # Create network, optimizer and loss
        self.conv_network = ConvNetwork(self.channels, X.shape[1], len(self.classes_))
        self.conv_network.train()

        optimizer: torch.optim.Optimizer
        if self.opt_type == OptType.LBGFS:
            optimizer = torch.optim.LBFGS(self.conv_network.parameters())
        elif self.opt_type == OptType.ADAM:
            optimizer = torch.optim.Adam(self.conv_network.parameters())
        else:
            optimizer = torch.optim.SGD(self.conv_network.parameters(), lr=0.1)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Make data utilities
        data = ExperimentData(X, self.inverse_classes_[y])
        if batch_size is None:
            if self.opt_type == OptType.LBGFS:
                # All dataset at once for LBGFS
                batch_size = len(data)
            else:
                batch_size = min(200, len(data))
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

        # Train
        for ep in range(epochs):
            self.training_epoch(ep, data_loader, optimizer, loss_fn)
        return self

    def training_epoch(self, ep, data_loader, optimizer, loss_fn):
        ct = 0
        running_err = 0
        for inp, target in data_loader:
            running_err += self.run_step(inp, target, optimizer, loss_fn)
            ct += 1
        if self.verbose:
            print(f"Epoch {ep}, error is {running_err / ct}")

    def run_step(self, inp, target, optimizer, loss_fn):
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            if self.conv_network is not None:
                out = self.conv_network.forward(inp.float())[:, :, 0]
            else:
                raise ValueError("Uninitialized network")
            err = loss_fn(out, target)
            if err.requires_grad:
                err.backward()
            return err

        optimizer.step(closure)
        return closure().item()

    def predict(self, X):
        if self.conv_network is not None:
            self.conv_network.eval()
        else:
            raise ValueError("Uninitialized network")
        check_is_fitted(self)

        out = self.conv_network.forward(torch.Tensor(X))[:, :, 0]
        return self.classes_[out.argmax(1).detach().numpy()]


class AdamClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.optimizer: Optional[optim.Adam] = None
        self.linear: Optional[nn.Linear] = None
        self.classes: Optional[np.ndarray] = None
        self.is_fitted = False
        self.loss_fn = nn.CrossEntropyLoss()

    def partial_fit(self, X, y, classes: np.ndarray = None) -> "AdamClassifier":
        if not self.is_fitted and classes is not None:
            self.classes = classes
            self.cls_map = {self.classes[i]: i for i in range(len(self.classes))}
            self.inv_cls_map = {v: k for (k, v) in self.cls_map.items()}
            self.linear = nn.Linear(X.shape[-1], len(self.classes))
            self.optimizer = optim.Adam(self.linear.parameters(), lr=0.01)
            self.is_fitted = True
        elif not self.is_fitted:
            raise ValueError(
                "Uninitialized classifer needs classes for "
                "the first call to partial_fit"
            )
        assert (
            self.linear is not None
            and self.optimizer is not None
            and self.classes is not None
        )

        self.linear.train()
        self.optimizer.zero_grad()
        out = self.linear(torch.Tensor(X))
        error = self.loss_fn(out, torch.Tensor([self.cls_map[i] for i in y]).long())
        error.backward()
        self.optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            self.linear.eval()
            return np.array(
                [
                    self.inv_cls_map[i]
                    for i in self.linear(torch.Tensor(X)).argmax(1).detach().numpy()
                ]
            )
        else:
            raise ValueError("Uninitialized network")


class CLSType(Enum):
    SGD = 1
    ADAM = 2


class SGDCls(BaseEstimator, ClassifierMixin):
    """A SGD-based linear classifier that can keep track of the testing loss
    during progress.

    """

    def __init__(self, verbose: bool = False, cls_type: CLSType = CLSType.SGD):
        self.verbose = verbose
        if cls_type == CLSType.SGD:
            self.sgd = SGDClassifier(
                learning_rate="constant", eta0=0.001, alpha=0.001, loss="log"
            )
        else:
            self.sgd = AdamClassifier()
        self.test_values: List[float] = []

    def fit(self, X, y, X_t=None, y_t=None, batch_size: int = 8) -> "SGDCls":
        self.test_values = []
        # Check classes
        self.classes_ = unique_labels(y)
        self.inverse_classes_ = np.zeros(self.classes_.max() + 1, dtype=int)
        for i, c in enumerate(self.classes_):
            self.inverse_classes_[c] = i

        classes = self.classes_
        self.sgd.partial_fit(X[0:1], y[0:1], classes=classes)
        if X_t is not None and y_t is not None:
            self.test_values.append(self.sgd.score(X_t, y_t))

        for i in range(0, X.shape[0], batch_size):
            batch_x, batch_y = X[i : i + batch_size], y[i : i + batch_size]
            self.sgd.partial_fit(batch_x, batch_y, classes=classes)
            if X_t is not None and y_t is not None:
                self.test_values.append(self.sgd.score(X_t, y_t))
                logging.debug("Validation score %s", self.test_values[-1])
            if classes is not None:
                classes = None
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.sgd.predict(X)

    def params(self) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.sgd, SGDClassifier):
            return (self.sgd.coef_, self.sgd.intercept_)
        elif self.sgd.linear is not None:
            return (
                self.sgd.linear.weight.data.numpy(),
                self.sgd.linear.bias.data.numpy(),
            )
        else:
            raise ValueError("Uninitialized classifier")
