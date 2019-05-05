import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from typing import Any, List, Tuple


def train_classifier(model: torch.nn.Module,
                     train_dataloader: DataLoader,
                     val_dataloader: DataLoader,
                     warmup: int = 2,
                     patience: int = 5,
                     max_epochs: int = 100) -> None:
    """Train the classifier

    Parameters
    ----------
    model
        The classifier to be trained
    train_dataloader:
        An iterator which returns batches of training images and labels from the
        training dataset
    val_dataloader:
        An iterator which returns batches of training images and labels from the
        validation dataset
    warmup: int, default: 2
        The number of epochs for which only the final layers (not from the ResNet base)
        should be trained
    patience: int, default: 5
        The number of epochs to keep training without an improvement in performance on the
        validation set before early stopping
    max_epochs: int, default: 100
        The maximum number of epochs to train for
    """

    best_state_dict = model.state_dict()
    best_val_auc_roc = 0.5
    patience_counter = 0
    for i in range(max_epochs):
        if i <= warmup:
            # we start by finetuning the model
            optimizer = torch.optim.Adam([pam for name, pam in
                                          model.named_parameters() if 'classifier' in name])
        else:
            # then, we train the whole thing
            optimizer = torch.optim.Adam(model.parameters())

        train_data, val_data = _train_classifier_epoch(model, optimizer, train_dataloader,
                                                       val_dataloader)
        if val_data[1] > best_val_auc_roc:
            best_val_auc_roc = val_data[1]
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping!")
                model.load_state_dict(best_state_dict)
                return None


def train_segmenter(model: torch.nn.Module,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    warmup: int = 2,
                    patience: int = 5,
                    max_epochs: int = 100) -> None:
    """Train the segmentation model

    Parameters
    ----------
    model
        The segmentation model to be trained
    train_dataloader:
        An iterator which returns batches of training images and masks from the
        training dataset
    val_dataloader:
        An iterator which returns batches of training images and masks from the
        validation dataset
    warmup: int, default: 2
        The number of epochs for which only the upsampling layers (not trained by the classifier)
        should be trained
    patience: int, default: 5
        The number of epochs to keep training without an improvement in performance on the
        validation set before early stopping
    max_epochs: int, default: 100
        The maximum number of epochs to train for
    """
    best_state_dict = model.state_dict()
    best_loss = 1
    patience_counter = 0
    for i in range(max_epochs):
        if i <= warmup:
            # we start by 'warming up' the final layers of the model
            optimizer = torch.optim.Adam([pam for name, pam in
                                          model.named_parameters() if 'pretrained' not in name])
        else:
            optimizer = torch.optim.Adam(model.parameters())

        train_data, val_data = _train_segmenter_epoch(model, optimizer, train_dataloader,
                                                      val_dataloader)
        if np.mean(val_data) < best_loss:
            best_loss = np.mean(val_data)
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping!")
                model.load_state_dict(best_state_dict)
                return None


def _train_classifier_epoch(model: torch.nn.Module,
                            optimizer: Optimizer,
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader
                            ) -> Tuple[Tuple[List[Any], float],
                                       Tuple[List[Any], float]]:

    t_losses, t_true, t_pred = [], [], []
    v_losses, v_true, v_pred = [], [], []
    model.train()
    for x, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds.squeeze(1), y)
        loss.backward()
        optimizer.step()
        t_losses.append(loss.item())

        t_true.append(y.cpu().detach().numpy())
        t_pred.append(preds.squeeze(1).cpu().detach().numpy())

    with torch.no_grad():
        model.eval()
        for val_x, val_y in tqdm(val_dataloader):
            val_preds = model(val_x)
            val_loss = F.binary_cross_entropy(val_preds.squeeze(1), val_y)
            v_losses.append(val_loss.item())

            v_true.append(val_y.cpu().detach().numpy())
            v_pred.append(val_preds.squeeze(1).cpu().detach().numpy())

    train_auc = roc_auc_score(np.concatenate(t_true), np.concatenate(t_pred))
    val_auc = roc_auc_score(np.concatenate(v_true), np.concatenate(v_pred))

    print(f'Train loss: {np.mean(t_losses)}, Train AUC ROC: {train_auc}, '
          f'Val loss: {np.mean(v_losses)}, Val AUC ROC: {val_auc}')
    return (t_losses, train_auc), (v_losses, val_auc)


def _train_segmenter_epoch(model: torch.nn.Module,
                           optimizer: Optimizer,
                           train_dataloader: DataLoader,
                           val_dataloader: DataLoader
                           ) -> Tuple[List[Any], List[Any]]:
    t_losses, v_losses = [], []
    model.train()
    for x, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        t_losses.append(loss.item())

    with torch.no_grad():
        model.eval()
        for val_x, val_y in tqdm(val_dataloader):
            val_preds = model(val_x)
            val_loss = F.binary_cross_entropy(val_preds, val_y.unsqueeze(1))
            v_losses.append(val_loss.item())
    print(f'Train loss: {np.mean(t_losses)}, Val loss: {np.mean(v_losses)}')

    return t_losses, v_losses
