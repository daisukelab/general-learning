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


def seed_everything(seed=42):
    """copied from dl-cliche"""
    if seed is None: return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, target='acc', objective='max', patience=10):
        self.crit_targ = target.lower()
        self.crit_obj = objective.lower()
        self.patience = patience
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
                status = True
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
            all_preds.extend(outputs.argmax(-1).detach().cpu().numpy())
        val_loss /= len(dl)
    if return_values:
        return val_loss, np.array(all_targets), np.array(all_preds)
    return val_loss


def _train(device, model, dl, criterion, optimizer, scheduler=None):
    model.train()
    train_loss = 0.0
    for inputs, labels in dl:
        inputs = inputs.to(device)
        org_labels = labels = labels.to(device)
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
    return results


def _train_model(device, model, criterion, optimizer, scheduler, trn_dl, val_dl, metric='acc',
                num_epochs=200, seed=None, patience=10, logger=None):
    seed_everything(seed)
    logger = logger if logger else logging.getLogger(__name__)
    early_stopper = EarlyStopping(patience=patience, target=metric, objective='max')
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
        logger.info(f'epoch {epoch+1:04d}/{num_epochs}: lr: {cur_lr:.7f}: loss={trn_loss:.6f} '
                    + ' '.join([f'val_{n}={v:.7f}' for n, v in val_metrics.items()]))
        # early stopping
        if early_stopper.on_epoch_end(epoch, model, val_metrics):
            break

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    for n, v in early_stopper.best_metrics.items():
        logger.info(f'Best val_{n}@{early_stopper.best_epoch+1} = {v}')

    # load best model weights
    model.load_state_dict(early_stopper.best_weights)
    return model, early_stopper.best_epoch, early_stopper.best_metrics


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        return out


class TorchMLPClassifier:
    """scikit-learn compatible PyToch based Multi-layer Perceptron.
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/_multilayer_perceptron.py


    """

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000, debug=False):
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
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def fit(self, X, y, device=torch.device('cuda'), logger=None):
        n_samples = len(X)
        bs = min(200, n_samples) if self.batch_size.lower() == 'auto' else self.batch_size
        train_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': self.shuffle}
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.classes = sorted(list(set(y)))
        Xtrn, Xval, ytrn, yval = train_test_split(X, y, test_size=self.validation_fraction, random_state=self.random_state)
        Xtrn, Xval, ytrn, yval = torch.Tensor(Xtrn), torch.Tensor(Xval), torch.tensor(ytrn), torch.tensor(yval)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtrn, ytrn), **train_kwargs)
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xval, yval), **test_kwargs)

        model = MLP(input_size=X.shape[-1], hidden_size=self.hidden_layer_sizes[0], output_size=len(self.classes))
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init, betas=(self.beta_1, self.beta_2), eps=self.epsilon)
        self.criterion = nn.CrossEntropyLoss()
        return _train_model(device, self.model, self.criterion, self.optimizer, None, train_loader, eval_loader, metric='acc',
                            num_epochs=self.max_iter, seed=self.random_state, patience=self.n_iter_no_change, logger=logger)

    def score(self, test_X, test_y, device=torch.device('cuda'), logger=None):
        logger = logger if logger else logging.getLogger(__name__)
        bs = 256 if self.batch_size.lower() == 'auto' else self.batch_size
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}

        test_X = self.scaler.transform(test_X)

        Xval, yval = torch.Tensor(test_X), torch.tensor(test_y)
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xval, yval), **test_kwargs)

        val_loss, targets, preds = _validate(device, self.model, eval_loader, self.criterion)
        metrics = _calc_metric(['acc'], targets, preds)
        return metrics['acc']
