from torch.utils.data import Dataset
from skimage import io


class BrainMRIDataset(Dataset):
    """
    image_dir (str): relative path to the images
    mask_dir (str: relative path to the masks
    transforms (albumentations.Compose): data transformation pipeline
    """

    def __init__(self, df, transform=None):
        self.image_paths = df['image_paths'].tolist()
        self.mask_path = df['mask_paths'].tolist()
        self.transforms = transform

    def __getitem__(self, i):
        image = io.imread(self.image_paths[i])
        mask = io.imread(self.mask_path[i])

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask.unsqueeze(0)

    def __len__(self):
        return len(self.image_paths)
