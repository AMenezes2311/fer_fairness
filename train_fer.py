# train_fer.py

import os
import argparse
from typing import Tuple, Optional, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, confusion_matrix

from datasets.fer2013_dataset import FER2013Dataset
from models.vgg_fer import VGGFER


# -----------------------------
# Transforms
# -----------------------------

def get_transforms(split: str):
    """
    Data transforms roughly matching the paper:
    - Random affine (rotate/shift/scale) with p=0.5 for training.
    - Random crop to 40x40 for training.
    - Validation/test: only ToTensor(); we crop inside evaluation.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=10,
                    translate=(0.2, 0.2),   # 20% shifts
                    scale=(0.8, 1.2),       # 20% rescale
                )],
                p=0.5
            ),
            transforms.ToTensor(),          # [0,1]
            transforms.RandomCrop(40),      # 40x40 crop
            transforms.RandomErasing(p=0.5)
        ])
    else:
        # keep full 48x48 tensor; we'll center-crop or ten-crop later
        return transforms.Compose([
            transforms.ToTensor(),
        ])


# -----------------------------
# Utilities
# -----------------------------

def center_crop_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    Center-crop a 4D tensor (B, C, H, W) to (size, size).
    If already that size, returns unchanged.
    """
    _, _, h, w = x.shape
    if h == size and w == size:
        return x
    top = (h - size) // 2
    left = (w - size) // 2
    return x[:, :, top:top + size, left:left + size]


# -----------------------------
# Training / evaluation loops
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in tqdm(loader, desc="Train", leave=False):
        if isinstance(batch, (list, tuple)):
            imgs, labels = batch[:2]
        else:
            imgs, labels = batch

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    crop_size: int = 40,
) -> Tuple[float, float, np.ndarray]:
    """
    Single-crop evaluation (center crop) for validation.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if isinstance(batch, (list, tuple)):
                imgs, labels = batch[:2]
            else:
                imgs, labels = batch

            imgs = imgs.to(device)
            labels = labels.to(device)

            # center-crop to 40x40 (if coming from 48x48)
            imgs = center_crop_tensor(imgs, crop_size)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, cm


def eval_with_tencrop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    crop_size: int = 40,
    num_classes: int = 7,
):
    """
    Ten-crop evaluation for test set:
    - 10 crops (4 corners + center + flipped) of size crop_size.
    - Average predictions over the 10 crops.
    Returns loss, accuracy, confusion matrix, indices, preds, labels, probs.
    """
    model.eval()
    running_loss = 0.0

    ten_crop = transforms.TenCrop(crop_size)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_indices: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test (10-crop)", leave=False):
            indices = None

            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    imgs, labels, indices = batch
                else:
                    imgs, labels = batch[:2]
            else:
                imgs, labels = batch

            b = imgs.size(0)
            labels = labels.to(device)

            # create 10 crops per image
            crops = torch.stack([
                torch.stack(ten_crop(img))  # (10, C, Hc, Wc)
                for img in imgs
            ])  # (B, 10, C, Hc, Wc)
            b, ncrops, c, h, w = crops.size()
            crops = crops.view(-1, c, h, w).to(device)  # (B*10, C, Hc, Wc)

            outputs = model(crops)                      # (B*10, num_classes)
            outputs = outputs.view(b, ncrops, num_classes).mean(1)  # (B, num_classes)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * b

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1)

            all_probs.append(probs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if indices is not None:
                all_indices.append(indices.numpy())
            else:
                # fall back to running range if dataset doesn't return indices
                start = len(all_labels) * b - b
                all_indices.append(np.arange(start, start + b))

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_indices = np.concatenate(all_indices)
    epoch_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, cm, all_indices, all_preds, all_labels, all_probs


# -----------------------------
# Main
# -----------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # datasets
    train_dataset = FER2013Dataset(
        csv_path=args.csv_path,
        split="Training",
        transform=get_transforms("train")
    )
    val_dataset = FER2013Dataset(
        csv_path=args.csv_path,
        split="PublicTest",
        transform=get_transforms("val")
    )
    test_dataset = FER2013Dataset(
        csv_path=args.csv_path,
        split="PrivateTest",
        transform=get_transforms("test")
    )

    # loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # model, loss, optimizer, scheduler
    model = VGGFER(num_classes=7, dropout_p=0.5).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,          # typically 0.01
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",          # we pass validation accuracy
        factor=0.75,
        patience=5
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    # training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc, val_cm = evaluate(
            model, val_loader, criterion, device, crop_size=40
        )
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New best model saved at epoch {epoch} (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # final test evaluation with ten-crop averaging
    test_loss, test_acc, test_cm, test_indices, test_preds, test_labels, test_probs = \
        eval_with_tencrop(model, test_loader, criterion, device, crop_size=40, num_classes=7)

    print(f"\nTest loss: {test_loss:.4f} | Test acc (10-crop): {test_acc:.4f}")
    print("Test confusion matrix:\n", test_cm)

    # save confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix_test.csv")
    pd.DataFrame(test_cm).to_csv(cm_path, index=False)
    print(f"Saved test confusion matrix to {cm_path}")

    # save per-sample predictions & probs for fairness analysis
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    data = {
        "index": test_indices,
        "true_label": test_labels,
        "pred_label": test_preds,
    }
    for cls in range(test_probs.shape[1]):
        data[f"prob_{cls}"] = test_probs[:, cls]

    pd.DataFrame(data).to_csv(pred_path, index=False)
    print(f"Saved test predictions to {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/fer2013.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)  # as in the paper
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
