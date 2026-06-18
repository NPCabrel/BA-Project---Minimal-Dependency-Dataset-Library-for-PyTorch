# ============================================================
# IMPORTS AND CONFIGURATION
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
from datetime import datetime

# Chemins
DATA_DIR = "/ds-sds/images/imagenet"
CHECKPOINT_DIR = "/netscratch/nague/resnet50_ddp_checkpoints"

# Paramètres d'entraînement
BATCH_SIZE = 256
NUM_EPOCHS = 90
LEARNING_RATE = 0.2  # Doublé car batch size effectif doublé (2×256=512)
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Logging
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 10
NUM_WORKERS = 8

# DDP Configuration
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

# ============================================================
# INITIALISATION DDP
# ============================================================
if WORLD_SIZE > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)

# ============================================================
# DATA LOADING
# ============================================================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if LOCAL_RANK == 0:
    print("Loading datasets...")

train_dataset = torchvision.datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=f"{DATA_DIR}/val", transform=val_transforms)

# Distributed Samplers
train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

if LOCAL_RANK == 0:
    print(f"Training Dataset: {len(train_dataset):,} images")
    print(f"Validation Dataset: {len(val_dataset):,} images")
    print(f"Classes: {len(train_dataset.classes)}")
    print(f"DDP: WORLD_SIZE={WORLD_SIZE}, LOCAL_RANK={LOCAL_RANK}")

# ============================================================
# MODEL
# ============================================================
if LOCAL_RANK == 0:
    print("Creating ResNet-50...")

model = models.resnet50(weights=None)
model = model.cuda(LOCAL_RANK)
if WORLD_SIZE > 1:
    model = DDP(model, device_ids=[LOCAL_RANK])

if LOCAL_RANK == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Model on GPU: {next(model.parameters()).is_cuda}")

# ============================================================
# LOSS AND OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if LOCAL_RANK == 0:
    print(f"Loss: {criterion}")
    print(f"Optimizer: SGD (lr={LEARNING_RATE}, momentum={MOMENTUM})")

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    loader.sampler.set_epoch(epoch)

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.cuda(LOCAL_RANK)
        labels = labels.cuda(LOCAL_RANK)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (predicted == labels).sum().item()
        total_samples += images.size(0)

        if LOCAL_RANK == 0 and (batch_idx + 1) % LOG_INTERVAL == 0:
            current_loss = running_loss / total_samples
            current_acc = 100.0 * running_corrects / total_samples
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {current_loss:.4f} Acc: {current_acc:.2f}% Time: {elapsed:.1f}s")

    epoch_loss = running_loss / total_samples
    epoch_acc = 100.0 * running_corrects / total_samples
    epoch_time = time.time() - start_time

    if LOCAL_RANK == 0:
        print(f"Epoch [{epoch+1}] ended - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% - Time: {epoch_time:.1f}s")
    return epoch_loss, epoch_acc

# ============================================================
# VALIDATION FUNCTION
# ============================================================
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda(LOCAL_RANK)
            labels = labels.cuda(LOCAL_RANK)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += (predicted == labels).sum().item()
            total_samples += images.size(0)

    val_loss = running_loss / total_samples
    val_acc = 100.0 * running_corrects / total_samples

    if LOCAL_RANK == 0:
        print(f"Validation - Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    return val_loss, val_acc

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
if LOCAL_RANK == 0:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("\n" + "="*60)
    print("DDP TRAINING STARTED")
    print("="*60)
    print(f"GPUs: {WORLD_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Effective batch size: {BATCH_SIZE * WORLD_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Initial LR: {LEARNING_RATE}")
    print("="*60 + "\n")

best_val_acc = 0.0
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    if LOCAL_RANK == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} (LR: {current_lr:.6f}) ---")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion)
    scheduler.step()

    if LOCAL_RANK == 0:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  -> Best model: {checkpoint_path} (Acc: {val_acc:.2f}%)")

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, periodic_path)
            print(f"  -> Checkpoint: {periodic_path}")

if LOCAL_RANK == 0:
    total_time = time.time() - train_start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    print("\n" + "="*60)
    print(f"TRAINING COMPLETED in {int(hours)}h {int(minutes)}min")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)

if WORLD_SIZE > 1:
    dist.destroy_process_group()
