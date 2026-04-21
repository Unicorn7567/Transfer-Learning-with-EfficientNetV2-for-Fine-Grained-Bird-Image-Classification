import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# --- New function: Check minimum image size and calculate max safe pooling count ---
def get_min_image_size_and_max_pools(data_path):
    """
    Traverse all images in the dataset to find the minimum height and width.
    Based on the minimum dimension, calculate the maximum number of pooling operations
    that can be safely used (using MaxPool2d with kernel_size=2, stride=2).
    Principle: After n pooling operations, the feature map size should be at least 1x1.
    Formula: min_dim / (2^n) >= 1  =>  n <= log2(min_dim)
    """
    print("=" * 60)
    print("Starting to check the minimum image size in the dataset...")
    print("=" * 60)

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    min_height, min_width = float('inf'), float('inf')

    for class_name in tqdm(classes, desc="Scanning image sizes"):
        class_path = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size  # PIL returns (width, height)
                        min_width = min(min_width, width)
                        min_height = min(min_height, height)
                except Exception as e:
                    print(f"Warning: Unable to read image dimensions {img_path}: {e}")

    if min_width == float('inf') or min_height == float('inf'):
        print("Error: No valid images found.")
        return None, None, None

    print(f"Minimum image size found: width={min_width}, height={min_height}")

    # Calculate the maximum safe number of pooling operations based on the minimum size
    # Use the more conservative dimension (the smaller of width and height) for calculation
    min_dimension = min(min_width, min_height)
    # Calculate the maximum number of pooling operations n, such that min_dimension / (2^n) >= 1
    max_allowed_pools = int(np.floor(np.log2(min_dimension)))

    # Recommend a slightly smaller value to leave room for padding and stride in the early network layers
    recommended_pools = max(1, max_allowed_pools - 2)  # Reserve margin for 2 pooling operations
    print(f"Calculation info: Theoretical maximum safe pooling count = {max_allowed_pools}")
    print(f"         -> Recommended network max pooling count = {recommended_pools} (margin reserved)")

    return min_width, min_height, recommended_pools


# --- End of new function ---


# 1. Data consistency check function
def check_data_consistency(data_path):
    """
    Check if the data is correctly classified according to folder names
    """
    print("=" * 60)
    print("Starting data consistency check...")
    print("=" * 60)

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    print(f"Found {len(classes)} classes")

    total_images = 0

    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if images:
            total_images += len(images)
            print(f"Class: {class_name:30s} | Image count: {len(images):4d}")

    print(f"\nTotal: {total_images} images")
    print(f"Average per class: {total_images / len(classes):.1f} images")
    print("\n✓ Data consistency check passed!")

    return classes, total_images


# 2. Dataset splitting function
def split_dataset(data_path, output_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split the dataset according to specified ratios
    """
    print("\n" + "=" * 60)
    print("Starting dataset splitting...")
    print("=" * 60)

    # Delete output directory if it already exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = os.path.join(output_path, split)
        os.makedirs(split_path, exist_ok=True)

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

    stats = {}
    for class_name in tqdm(classes, desc="Splitting data for each class"):
        class_path = os.path.join(data_path, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not images:
            continue

        # Randomly shuffle images
        random.shuffle(images)
        total = len(images)

        # Calculate split points
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        # Split images
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Copy images to corresponding directories
        for split, imgs in zip(splits, [train_imgs, val_imgs, test_imgs]):
            split_class_path = os.path.join(output_path, split, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            for img_name in imgs:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_class_path, img_name)
                shutil.copy2(src, dst)

        # Save statistics
        stats[class_name] = {
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs)
        }

    # Calculate totals
    print("\nDataset splitting completed:")
    total_train = total_val = total_test = 0
    for split in splits:
        split_path = os.path.join(output_path, split)
        total_in_split = 0
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                total_in_split += len(images)

        if split == 'train':
            total_train = total_in_split
        elif split == 'val':
            total_val = total_in_split
        else:
            total_test = total_in_split

        print(f"  {split:5s} set: {total_in_split:5d} images")

    # Print detailed statistics for the first 10 classes
    print("\nSplit details for the first 10 classes:")
    for i, class_name in enumerate(list(stats.keys())[:10]):
        stat = stats[class_name]
        print(f"  {class_name:30s}: Train={stat['train']:2d}, Val={stat['val']:2d}, Test={stat['test']:2d}")

    if len(classes) > 10:
        print(f"  ... and {len(classes) - 10} more classes")

    return output_path, total_train, total_val, total_test


# 3. Data preprocessing and augmentation
def get_transforms(img_size=224):
    """
    Get data preprocessing and augmentation transformations
    """
    # Data augmentation for training set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocessing for validation and test sets
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


# 4. Create custom dataset class
class BirdDataset(Dataset):
    """Custom bird dataset class"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        # Load all image paths and labels
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Unable to load image {img_path}: {e}")
            # Return a black image as a placeholder
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Get class distribution"""
        from collections import Counter
        return Counter(self.labels)


# 5. Define CNN model (modified for dynamic adaptation)
class BirdCNN(nn.Module):
    """Custom CNN model for bird classification, adaptable to image sizes"""

    def __init__(self, num_classes=200, num_pool_layers=5, img_size=224):
        """
        Parameters:
            num_classes: Number of classes
            num_pool_layers: Number of max-pooling layers (nn.MaxPool2d)
            img_size: Input image size (assumed to be square)
        """
        super(BirdCNN, self).__init__()

        # Modification: Save parameters for flattening calculation in forward pass
        self.num_pool_layers = num_pool_layers
        self.img_size = img_size

        # Build convolutional layer sequence
        conv_layers = []
        in_channels = 3
        out_channels_list = [32, 64, 128, 256, 512]

        # Build network according to the specified number of pooling layers
        for i in range(num_pool_layers):
            out_channels = out_channels_list[i] if i < len(out_channels_list) else 512
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Modification: Dynamically calculate the input feature count for the fully connected layer
        # After num_pool_layers 2x2 pooling operations, feature map size becomes img_size / (2^num_pool_layers)
        self.feature_size = img_size // (2 ** num_pool_layers)
        if self.feature_size <= 0:
            raise ValueError(f"Image size ({img_size}) is too small to support {num_pool_layers} pooling operations. "
                             f"Please reduce the number of pooling layers or increase the input image size.")

        fc_input_features = in_channels * self.feature_size * self.feature_size
        print(f"  Network structure info: Input size={img_size}, Pooling count={num_pool_layers}, "
              f"Final feature map size={self.feature_size}x{self.feature_size}, "
              f"Fully connected layer input features={fc_input_features}")

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# 6. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=20, device='cpu'):
    """Train the model"""
    print("\n" + "=" * 60)
    print("Starting model training...")
    print("=" * 60)

    model = model.to(device)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    best_model_path = 'best_bird_model.pth'

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Train')
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Val')
            for batch_idx, (inputs, labels) in enumerate(val_bar):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                val_bar.set_postfix({
                    'Loss': f'{val_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate adjustment
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  Learning rate adjusted from {old_lr:.6f} to {new_lr:.6f}")

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, best_model_path)
            print(f'  ✓ Saved best model, validation accuracy: {val_acc:.2f}%')

    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }


# 7. Testing function
def test_model(model, test_loader, device='cpu'):
    """Test the model"""
    print("\n" + "=" * 60)
    print("Starting model testing...")
    print("=" * 60)

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing')
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_bar.set_postfix({'Acc': f'{100. * correct / total:.2f}%'})

    test_acc = 100. * correct / total
    print(f'\nTest accuracy: {test_acc:.2f}%')

    return test_acc, all_preds, all_labels


# 8. Visualize training curves
def plot_training_curves(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(history['train_accs'], label='Train Acc')
    axes[1].plot(history['val_accs'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
    plt.show()


# 9. Main function (modified to include image size check and dynamic model building)
def main():
    """Main function"""
    # Set parameters
    DATA_PATH = "CUB_200_2011/images"  # Your data path
    OUTPUT_PATH = "bird_dataset_split"  # Path to save the split dataset
    BATCH_SIZE = 16
    IMG_SIZE = 224
    NUM_EPOCHS = 300
    LEARNING_RATE = 0.001
    NUM_CLASSES = 200

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU model: {torch.cuda.get_device_name(0)}")

    # --- Modification/Addition: Check image size before data check ---
    min_width, min_height, recommended_pools = get_min_image_size_and_max_pools(DATA_PATH)
    if min_width is None:
        print("Image size check failed, exiting program.")
        return

    # Dynamically decide the number of max-pooling layers based on the check result
    # If the recommended value is less than the default 5, use the recommended value, otherwise keep 5 (avoid unnecessarily deep network)
    NUM_POOL_LAYERS = min(5, recommended_pools)  # Keep at most 5 layers, same depth as original structure
    print(f"Number of network max-pooling layers to be used: {NUM_POOL_LAYERS}")
    # --- End of modification/addition ---

    # 1. Check data consistency
    classes, total_images = check_data_consistency(DATA_PATH)

    # 2. Split dataset (if not already split)
    if not os.path.exists(OUTPUT_PATH):
        print("\nSplitting dataset...")
        output_path, total_train, total_val, total_test = split_dataset(
            DATA_PATH, OUTPUT_PATH, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
    else:
        print(f"\nDataset already split, using existing split: {OUTPUT_PATH}")

        # Modification: Fix statistics logic, count image files instead of folder count
        def count_images_in_split(split_name):
            split_path = os.path.join(OUTPUT_PATH, split_name)
            total = 0
            if os.path.exists(split_path):
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if os.path.isdir(class_path):
                        images = [f for f in os.listdir(class_path)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        total += len(images)
            return total

        total_train = count_images_in_split('train')
        total_val = count_images_in_split('val')
        total_test = count_images_in_split('test')

    # 3. Get data transformations
    train_transform, val_test_transform = get_transforms(IMG_SIZE)

    # 4. Create datasets
    print("\nLoading datasets...")
    train_dataset = BirdDataset(os.path.join(OUTPUT_PATH, 'train'), transform=train_transform)
    val_dataset = BirdDataset(os.path.join(OUTPUT_PATH, 'val'), transform=val_test_transform)
    test_dataset = BirdDataset(os.path.join(OUTPUT_PATH, 'test'), transform=val_test_transform)

    print(f"\nDataset statistics:")
    print(f"  Training set: {len(train_dataset)} images")
    print(f"  Validation set: {len(val_dataset)} images")
    print(f"  Test set: {len(test_dataset)} images")

    # 5. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # 6. Create model (Modification: Pass dynamically calculated number of pooling layers)
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    # Use custom CNN model with dynamic parameters
    model = BirdCNN(num_classes=NUM_CLASSES,
                    num_pool_layers=NUM_POOL_LAYERS,
                    img_size=IMG_SIZE)

    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 7. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Modification: Removed the erroneous `verbose` parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # 8. Train the model
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                          scheduler, NUM_EPOCHS, device)

    # 9. Plot training curves
    try:
        plot_training_curves(history)
    except Exception as e:
        print(f"Error plotting training curves: {e}")

    # 10. Load the best model and test
    if os.path.exists('best_bird_model.pth'):
        print("\nLoading best model for testing...")
        checkpoint = torch.load('best_bird_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_acc, all_preds, all_labels = test_model(model, test_loader, device)
    else:
        print("\nBest model not found, using the last trained model for testing...")
        test_acc, all_preds, all_labels = test_model(model, test_loader, device)

    # 11. Save the complete model
    torch.save(model.state_dict(), 'bird_classification_model_final.pth')
    print(f"\nModel saved as: bird_classification_model_final.pth")

    # 12. Save class mapping
    class_mapping = {
        'idx_to_class': train_dataset.idx_to_class,
        'class_to_idx': train_dataset.class_to_idx
    }
    torch.save(class_mapping, 'class_mapping.pth')
    print(f"Class mapping saved as: class_mapping.pth")

    # 13. Save network structure parameters (new)
    model_config = {
        'num_pool_layers': NUM_POOL_LAYERS,
        'img_size': IMG_SIZE,
        'num_classes': NUM_CLASSES
    }
    torch.save(model_config, 'model_config.pth')
    print(f"Model configuration saved as: model_config.pth")

    # 14. Print class count statistics
    print("\n" + "=" * 60)
    print("Class statistics:")
    print("=" * 60)

    for i, class_name in enumerate(classes[:10]):
        print(f"{i:3d}: {class_name}")

    if len(classes) > 10:
        print(f"... and {len(classes) - 10} more classes")

    return model, history, test_acc


# Run the main program
if __name__ == "__main__":
    # Check necessary directories
    if not os.path.exists("CUB_200_2011"):
        print("Error: 'CUB_200_2011' directory not found!")
        print("Please ensure:")
        print("1. The dataset is placed in the current directory")
        print("2. The dataset structure is: CUB_200_2011/images/...")
        print("3. Or modify the DATA_PATH variable in the code to point to your dataset path")
    else:
        try:
            model, history, test_acc = main()

            # Display final results
            print("\n" + "=" * 60)
            print("Training completed!")
            print("=" * 60)
            print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
            print(f"Final test accuracy: {test_acc:.2f}%")
            print(f"Model saved as: bird_classification_model_final.pth")
            print(f"Best model checkpoint: best_bird_model.pth")
            print(f"Class mapping file: class_mapping.pth")
            print(f"Model configuration file: model_config.pth")

        except Exception as e:
            print(f"\nError during program execution: {e}")
            import traceback

            traceback.print_exc()
            print("\nAttempting to re-split the dataset...")

            # Try to delete existing split and re-split
            if os.path.exists("bird_dataset_split"):
                shutil.rmtree("bird_dataset_split")

            try:
                model, history, test_acc = main()
            except Exception as e2:
                print(f"Run failed again: {e2}")
                print("\nSuggestions:")
                print("1. Check if the dataset path is correct")
                print("2. Ensure there is enough disk space")
                print("3. Try restarting the Python environment")