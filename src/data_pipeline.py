import os
import torch
from torchvision import datasets
from torch.utils.data import random_split
from PIL import Image
from tqdm import tqdm

def prepare_cifar10_data(data_root="data"):
    raw_dir = os.path.join(data_root, "raw")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # Folders ensure karein
    for d in [raw_dir, train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    print("Downloading CIFAR-10 dataset via torchvision...")
    full_trainset = datasets.CIFAR10(root=raw_dir, train=True, download=True)
    testset = datasets.CIFAR10(root=raw_dir, train=False, download=True)

    # 80-20 split on training data (40,000 train, 10,000 val)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_subset, val_subset = random_split(
        full_trainset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    class_names = full_trainset.classes

    def export_images(dataset_subset, target_folder, is_subset=True):
        for class_name in class_names:
            os.makedirs(os.path.join(target_folder, class_name), exist_ok=True)
        
        print(f"Exporting images to {target_folder}...")
        for idx in tqdm(range(len(dataset_subset))):
            img, label = dataset_subset[idx]
            class_name = class_names[label]
            file_path = os.path.join(target_folder, class_name, f"img_{idx:05d}.png")
            img.save(file_path)

    # Images export to folders
    export_images(train_subset, train_dir, is_subset=True)
    export_images(val_subset, val_dir, is_subset=True)
    export_images(testset, test_dir, is_subset=False)

    print("\nDataset preparation complete:")
    print(f"- Train set: {len(train_subset)} images in {train_dir}")
    print(f"- Val set:   {len(val_subset)} images in {val_dir}")
    print(f"- Test set:  {len(testset)} images in {test_dir}")

if __name__ == "__main__":
    prepare_cifar10_data()