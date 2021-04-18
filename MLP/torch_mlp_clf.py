"""
PyToch based Multi-Layer Perceptron Classifier, compatible interface with scikit-learn.
Using GPU by default to run faster.

Disclimer:
    NOT FULLY COMPATIBLE w/ scikit-learn.

Reference:
    - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    - https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/_multilayer_perceptron.py
"""

import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


def seed_everything(seed=42):
    if seed is None: return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_array_like(item):
    """Check if item is an array-like object."""
    return isinstance(item, (list, set, tuple, np.ndarray))


def all_same_classes(y_a, y_b, delimiter=None):
    """Test if all classes in y_a is also in y_b or not.
    If y_a is a single dimension array, test as single labeled.
    If y_a is a two dimension array, test as multi-labeled.

    Args:
        y_a: One list of labels.
        y_b: Another list of labels.
        delimiter: Set a character if multi-label text is given.

    Returns:
        True or False.
    """
    if is_array_like(y_a[0]):
        # binary matrix multi-label table, test that class existance is the same.
        y_a, y_b = y_a.sum(axis=0), y_b.sum(axis=0)
        classes_a, classes_b = y_a > 0, y_b > 0
        return np.all(classes_a == classes_b)

    # test: classes contained in both array is consistent.
    if delimiter is not None:
        y_a = flatten_list([y.split(delimiter) for y in y_a])
        y_b = flatten_list([y.split(delimiter) for y in y_b])
    classes_a, classes_b = list(set(y_a)), list(set(y_b))
    return len(classes_a) == len(classes_b)


def train_test_sure_split(X, y, n_attempt=100, return_last=False, debug=False, **kwargs):
    """Variant of train_test_split that makes validation for sure.
    Returned y_test should contain all class samples at least one.
    Simply try train_test_split repeatedly until the result satisfies this condition.

    Args:
        n_attempt: Number of attempts to satisfy class coverage.
        return_last: Return last attempt results if all attempts didn't satisfy.

    Returns:
        X_train, X_test, y_train, y_test if satisfied;
        or None, None, None, None.
    """

    for i in range(n_attempt):
        X_trn, X_val, y_trn, y_val = train_test_split(X, y, **kwargs)
        if all_same_classes(y, y_val):
            return X_trn, X_val, y_trn, y_val
        if debug:
            print('.', end='')
    if return_last:
        return X_trn, X_val, y_trn, y_val
    return None, None, None, None


class EarlyStopping():
    def __init__(self, target='acc', objective='max', patience=10, enable=True):
        self.crit_targ = target
        self.crit_obj = objective
        self.patience = patience
        self.enable = enable
        self.stopped_epoch = 0
        self.wait = 0
        self.best_value = 0 if objective == 'max' else 1e15
        self.best_epoch = None
        self.best_weights = None
        self.best_metrics = None

    def on_epoch_end(self, epoch, model, val_metrics):
        status = False
        condition = (val_metrics[self.crit_targ] >= self.best_value
                     if self.crit_obj == 'max' else
                     val_metrics[self.crit_targ] <= self.best_value)
        if condition:
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_metrics = val_metrics
            self.best_value = val_metrics[self.crit_targ]
            self.wait = 1
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                status = self.enable
            self.wait += 1
        return status


def _validate(device, model, dl, criterion, return_values=True):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in dl:
            all_targets.extend(targets.numpy())
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets) * inputs.size(0)
            if targets.dim() == 1:
                outputs = outputs.softmax(-1).argmax(-1)
            elif targets.dim() == 2:
                outputs = outputs.sigmoid()
            all_preds.extend(outputs.detach().cpu().numpy())
        val_loss /= len(dl)
    if return_values:
        return val_loss, np.array(all_targets), np.array(all_preds)
    return val_loss


def _train(device, model, dl, criterion, optimizer, scheduler=None):
    model.train()
    train_loss = 0.0
    for inputs, labels in dl:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    if scheduler:
        scheduler.step()
    train_loss /= len(dl)
    return train_loss


def _calc_metric(metrics, targets, preds):
    results = {}
    if 'acc' in metrics:
        results['acc'] = sum(targets == preds) / len(targets)
    if 'mAP' in metrics:
        results['mAP'] = average_precision_score(targets, preds)
    return results


def _train_model(device, model, criterion, optimizer, scheduler, trn_dl, val_dl, metric='acc',
                num_epochs=200, seed=None, patience=10, stop_metric=None,
                early_stopping=False, logger=None):
    seed_everything(seed)
    logger = logger if logger else logging.getLogger(__name__)
    stop_metric = metric if stop_metric is None else stop_metric
    stop_objective = 'min' if stop_metric == 'loss' else 'max'
    early_stopper = EarlyStopping(patience=patience, target=stop_metric, objective=stop_objective,
                                  enable=early_stopping)
    since = time.time()

    for epoch in range(num_epochs):
        # train
        trn_loss = _train(device, model, trn_dl, criterion, optimizer, scheduler)
        # validate, calculate metrics
        val_loss, val_targets, val_preds = _validate(device, model, val_dl, criterion)
        val_metrics = _calc_metric([metric], val_targets, val_preds)
        val_metrics['loss'] = val_loss
        # print log
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.debug(f'epoch {epoch+1:04d}/{num_epochs}: lr: {cur_lr:.7f}: loss={trn_loss:.6f} '
                    + ' '.join([f'val_{n}={v:.7f}' for n, v in val_metrics.items()]))
        # early stopping
        if early_stopper.on_epoch_end(epoch, model, val_metrics):
            break

    time_elapsed = time.time() - since
    logger.debug(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    for n, v in early_stopper.best_metrics.items():
        logger.debug(f'Best val_{n}@{early_stopper.best_epoch+1} = {v}')

    # load best model weights
    model.load_state_dict(early_stopper.best_weights)
    return model, early_stopper.best_epoch, early_stopper.best_metrics


class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        fcs = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            fcs.append(nn.Linear(in_size, out_size))
            # fcs.append(nn.Dropout(0.2))
            fcs.append(nn.ReLU())
        self.mlp = nn.Sequential(*fcs[:-1])
        
    def forward(self, x):
        out = self.mlp(x)
        return out


class TorchMLPClassifier:

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=1e-8, # alpha=0.0001 --- too big for this implementation
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                 # Extra options
                 scaling=True, debug=False):
        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.alpha=alpha
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.learning_rate_init=learning_rate_init
        self.power_t=power_t
        self.max_iter=max_iter
        self.loss='log_loss'
        self.shuffle=shuffle
        self.random_state=random_state
        self.tol=tol
        self.verbose=verbose
        self.warm_start=warm_start
        self.momentum=momentum
        self.nesterovs_momentum=nesterovs_momentum
        self.early_stopping=early_stopping
        self.validation_fraction=validation_fraction
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon
        self.n_iter_no_change=n_iter_no_change
        self.max_fun=max_fun
        self.scaling = scaling
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def switch_regime(self, y):
        if y.ndim == 2: # multi label
            n_class = y.shape[1]
            return 'mAP', nn.BCEWithLogitsLoss(), n_class, torch.Tensor
        elif y.ndim == 1: # classification
            n_class = len(list(set(y)))
            return 'acc', nn.CrossEntropyLoss(), n_class, torch.tensor
        raise Exception(f'Unsupported shape of y: {y.shape}')

    def fit(self, X, y, X_val=None, y_val=None, device=None, logger=None, val_idxs=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y = np.array(y)
        metric, criterion, n_class, label_type = self.switch_regime(y)
        logger = logger if logger else logging.getLogger(__name__)

        n_samples = len(X)
        bs = min(200, n_samples) if self.batch_size.lower() == 'auto' else self.batch_size
        train_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': self.shuffle}
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}

        if self.scaling:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        if X_val is not None:
            Xtrn, Xval, ytrn, yval = X, X_val, y, y_val
        elif val_idxs is None:
            Xtrn, Xval, ytrn, yval = train_test_sure_split(X, y, test_size=self.validation_fraction,
                                                        random_state=self.random_state)
        else:
            mask = np.array([i in val_idxs for i in range(len(X))])
            Xtrn, Xval, ytrn, yval = X[~mask], X[mask], y[~mask], y[mask]
        Xtrn, Xval, ytrn, yval = torch.Tensor(Xtrn), torch.Tensor(Xval), label_type(ytrn), label_type(yval)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtrn, ytrn), **train_kwargs)
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xval, yval), **test_kwargs)

        model = MLP(input_size=X.shape[-1], hidden_sizes=self.hidden_layer_sizes, output_size=n_class)
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init,
                                          betas=(self.beta_1, self.beta_2), eps=self.epsilon, weight_decay=self.alpha)
        self.criterion = criterion
        if self.debug:
            print('Training model:', model)
            print('Details - metric:', metric, ' loss:', criterion, ' n_class:', n_class)
        return _train_model(device, self.model, self.criterion, self.optimizer, None, train_loader, eval_loader, metric=metric,
                            num_epochs=self.max_iter, seed=self.random_state, patience=self.n_iter_no_change,
                            early_stopping=self.early_stopping, logger=logger)

    def score(self, test_X, test_y, device=torch.device('cuda'), logger=None):
        test_y = np.array(test_y)
        metric, criterion, n_class, label_type = self.switch_regime(test_y)
        logger = logger if logger else logging.getLogger(__name__)

        bs = 256 if self.batch_size.lower() == 'auto' else self.batch_size
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}

        if self.scaling:
            test_X = self.scaler.transform(test_X)

        Xval, yval = torch.Tensor(test_X), label_type(test_y)
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xval, yval), **test_kwargs)

        val_loss, targets, preds = _validate(device, self.model, eval_loader, self.criterion)
        metrics = _calc_metric([metric], targets, preds)
        return metrics[metric]

    def predict(self, X, device=torch.device('cuda'), multi_label_n_class=None, logger=None):
        logger = logger if logger else logging.getLogger(__name__)

        bs = 256 if self.batch_size.lower() == 'auto' else self.batch_size
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}
        if self.scaling:
            X = self.scaler.transform(X)
        X = torch.Tensor(X)
        y = (torch.zeros((len(X)), dtype=torch.int) if multi_label_n_class is None else
             torch.zeros((len(X), multi_label_n_class), dtype=torch.float))
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), **test_kwargs)

        val_loss, targets, preds = _validate(device, self.model, eval_loader, self.criterion)
        return preds
