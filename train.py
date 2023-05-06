import torch
import albumentations as A
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from models import UNet
from loss import DiceLoss
from metric import IoU
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders
)

# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
DATA_DIR = 'data/lgg-mri-segmentation/kaggle_3m'


def train_fn(epoch_index, loader, model, optimizer, loss_fn, metric, scaler, tb_writer):
    running_loss = 0.
    running_iou = 0.
    last_loss = 0.
    last_iou = 0.
    loop = tqdm(loader)

    for batch_idx, (image, mask) in enumerate(loop):
        image = image.to(device=DEVICE, dtype=torch.float)
        mask = mask.to(device=DEVICE, dtype=torch.float)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(image)
            loss = loss_fn(predictions, mask)
            iou = metric(predictions, mask)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Gather data and report for every 1000 batches
        running_loss += loss.item()
        running_iou += iou.item()
        if batch_idx % 2 == 0:
            last_loss = running_loss / len(loader)
            last_iou = running_iou / len(loader)
            print('   batch {} loss: {:.4f} iou: {:.4f}'.format(batch_idx, last_loss, last_iou))
            tb_x = epoch_index * len(loader) + batch_idx + 1
            tb_writer.add_scalar('dice_loss/train', last_loss, tb_x)
            tb_writer.add_scalar('iou/train', last_iou, tb_x)
            running_loss = 0.
            running_iou = 0.

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return last_loss, last_iou


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0,
                rotate_limit=15,
                p=0.1
            ),
            A.OneOf([
                A.ElasticTransform(p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.8),
            A.Normalize(mean=(0.0921, 0.0834, 0.0877),
                        std=(0.1356, 0.1237, 0.1292)),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=(0.0921, 0.0834, 0.0877),
                        std=(0.1356, 0.1237, 0.1292)),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metric = IoU()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/BrainSegmentation/brain_segmentation_trainer_{}'.format(timestamp))
    best_vloss = 1_000_000

    train_loader, val_loader = get_loaders(
        DATA_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    # Start the timer
    start_time = timer()
    for epoch in range(NUM_EPOCHS):
        avg_loss, avg_iou = train_fn(epoch, train_loader, model, optimizer, loss_fn, metric, scaler, writer)

        model.eval()
        running_vloss = 0.0
        running_viou = 0.0
        with torch.no_grad():
            for i, (image, mask) in enumerate(val_loader):
                image = image.to(DEVICE, dtype=torch.float)
                mask = mask.to(DEVICE, dtype=torch.float)
                pred_mask = model(image)
                vloss = loss_fn(pred_mask, mask)
                viou = metric(pred_mask, mask)
                running_vloss += vloss
                running_viou += viou
        avg_vloss = running_vloss / (i + 1)
        avg_viou = running_viou / (i + 1)
        print(f'train: dice_loss - {avg_loss}  iou_score - {avg_iou}\nvalid: dice_loss - {avg_vloss}  iou_score - {avg_viou}')
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.add_scalars('Training vs. Validation IoU',
                           {'Training': avg_iou, 'Validation': avg_viou},
                           epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, timestamp, epoch)

    # End the timer and print out how long it took
    end_time = timer()
    print(f'Total training time: {end_time - start_time:.3f} seconds')


if __name__ == "__main__":
    main()
