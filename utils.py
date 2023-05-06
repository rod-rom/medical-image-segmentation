import torch
import torchvision
import pandas as pd
import numpy as np
from glob import glob
from skimage import io
from torch.utils.data import DataLoader
from dataset import BrainMRIDataset
from sklearn.model_selection import train_test_split


def create_dataframe(data_dir):
    image_paths = []
    mask_paths = glob(f'{data_dir}/*/*_mask*')
    for i in mask_paths:
        image_paths.append(i.replace('_mask', ''))
    df = pd.DataFrame(data={'image_paths': image_paths, 'mask_paths': mask_paths})
    df['diagnosis'] = df['mask_paths'].apply(lambda x: positive_negative_diagnosis(x))
    return df


def split_dataset(df, data_dir):
    # Split df into train_df and val_df
    train_df, val_df = train_test_split(create_dataframe(data_dir=data_dir), stratify=df.diagnosis, test_size=0.1,
                                        random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Split train_df into train_df and test_df
    train_df, test_df = train_test_split(train_df, stratify=train_df.diagnosis, test_size=0.15, random_state=42)
    train_df = train_df.reset_index(drop=True)
    return train_df, val_df, test_df


def positive_negative_diagnosis(mask_path):
    value = np.max(io.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def save_checkpoint(state, timestamp, epoch_number):
    print("=> Saving checkpoint")
    torch.save(state, 'model_{}_{}.pt'.format(timestamp, epoch_number))


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        data_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    df = create_dataframe(data_dir)
    train_df, val_df, test_df = split_dataset(df, data_dir)

    train_ds = BrainMRIDataset(
        df=train_df,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BrainMRIDataset(
        df=val_df,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, criterion, metric, device="cuda"):
    running_val_loss = []
    running_val_iou = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred_mask = model(x)
            loss = criterion(pred_mask, y)
            iou = metric(pred_mask, y)
            running_val_loss.append(loss.item())
            running_val_iou.append(iou.item())


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
