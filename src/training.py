from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:  # pragma: no cover - depends on the local scientific Python stack
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:  # pragma: no cover - fallback for incompatible scipy/sklearn wheels
    def roc_curve(y_true, y_score):
        y_true = np.asarray(y_true).astype(int).ravel()
        y_score = np.asarray(y_score, dtype=float).ravel()

        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_true = y_true[desc_score_indices]
        y_score = y_score[desc_score_indices]

        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[np.inf, y_score[threshold_idxs]]

        positives = max(np.sum(y_true), 1)
        negatives = max(y_true.size - np.sum(y_true), 1)
        return fps / negatives, tps / positives, thresholds

    def roc_auc_score(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.trapezoid(tpr, fpr))


def train(
    model,
    data_train,
    parameters_train,
    labels_train,
    data_validation,
    parameters_validation,
    labels_validation,
    config,
    device,
    show_output=True,
):
    model.to(device)

    assert isinstance(data_train, torch.Tensor)
    assert isinstance(parameters_train, torch.Tensor)
    assert isinstance(labels_train, torch.Tensor)
    assert data_train.shape[0] == parameters_train.shape[0] == labels_train.shape[0]
    assert isinstance(data_validation, torch.Tensor)
    assert isinstance(parameters_validation, torch.Tensor)
    assert isinstance(labels_validation, torch.Tensor)
    assert (
        data_validation.shape[0]
        == parameters_validation.shape[0]
        == labels_validation.shape[0]
    )

    train_dataset = torch.utils.data.TensorDataset(
        data_train,
        parameters_train,
        labels_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.BCELoss()

    results = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
    }

    for epoch in range(config["training"]["n_epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_data, batch_parameters, batch_labels in train_loader:
            batch_labels = batch_labels.unsqueeze(1)
            batch_labels, batch_data, batch_parameters = [
                x.to(device) for x in (batch_labels, batch_data, batch_parameters)
            ]
            inputs = torch.cat((batch_data, batch_parameters), dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        results["train_loss"].append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_labels = labels_validation.unsqueeze(1)
            val_labels, val_data, val_parameters = [
                x.to(device) for x in (val_labels, data_validation, parameters_validation)
            ]
            val_inputs = torch.cat((val_data, val_parameters), dim=1)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_accuracy = ((val_outputs >= 0.5).float() == val_labels).float().mean()
            val_auc = roc_auc_score(
                val_labels.detach().cpu().numpy(),
                val_outputs.detach().cpu().numpy(),
            )

        results["val_loss"].append(val_loss.item())
        results["val_accuracy"].append(val_accuracy.item())
        results["val_auc"].append(val_auc)
        if show_output:
            print(
                f"Epoch {epoch + 1}/{config['training']['n_epochs']}, "
                f"Loss: {results['train_loss'][-1]}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"Val Acc: {val_accuracy.item():.4f}, "
                f"Val AUC: {val_auc:.4f}"
            )

    return model, results


def test_model(model, data_test, parameters_test, labels_test, device, show_output=True):
    with torch.no_grad():
        model.eval()
        labels_test = labels_test.unsqueeze(1)
        labels_test, data_test, parameters_test = [
            x.to(device) for x in (labels_test, data_test, parameters_test)
        ]
        inputs = torch.cat((data_test, parameters_test), dim=1)
        outputs = model(inputs)
        test_loss = nn.BCELoss()(outputs, labels_test)
        test_accuracy = ((outputs >= 0.5).float() == labels_test).float().mean()
        test_auc = roc_auc_score(
            labels_test.detach().cpu().numpy(),
            outputs.detach().cpu().numpy(),
        )
        test_roc = roc_curve(
            labels_test.detach().cpu().numpy(),
            outputs.detach().cpu().numpy(),
        )

    if show_output:
        print(
            f"Test Loss: {test_loss.item():.4f}, "
            f"Test Acc: {test_accuracy.item():.4f}, "
            f"Test AUC: {test_auc:.4f}"
        )
    return test_loss.item(), test_accuracy.item(), test_auc, test_roc
