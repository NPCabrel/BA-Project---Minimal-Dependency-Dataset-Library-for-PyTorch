#==================================
#Importations and Initialisations
#==================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import time
from datetime import datetime

DATA_DIR = "/ds-sds/images/imagenet"    #Already downloaded
CHECKPOINT_DIR = "/netscratch/nague/resnet50_checkpoints"      #Saving models




# Training Parameters

BATCH_SIZE = 256           #our RTXA6000 has 48Go VRAM and ResNet has images of 224x224 ~ 4Go per batch. It's also possible to go up to 512, but 256 is sure
NUM_EPOCHS = 90
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4     #Avoid overfitting


# Logging Parameters
LOG_INTERVAL = 100          #Show Progression after 100 Batches
CHECKPOINT_INTERVAL = 10    # Save the model after each 10 epochs
NUM_WORKERS = 8        


#====================
#Data loading 
#====================


# Data Augmentation
train_transforms = transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(
		mean = [0.485, 0.456, 0.406],    #Mean ImageNet 
		std = [0.229, 0.224, 0.225])])     # Ecart-type


# Validation Tranformations (No Augmentation)
val_transforms = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
		mean = [0.485, 0.456, 0.406],
		std = [0.229, 0.224,  0.225])])

# Datasets creation
print ("train dataset loading .. ")
train_dataset = torchvision.datasets.ImageFolder(
	root=f"{DATA_DIR}/train",
	transform=train_transforms)

print ("Validation dataset loading .. ")
val_dataset = torchvision.datasets.ImageFolder(
	root=f"{DATA_DIR}/val",
	transform=val_transforms)

# Dataloaders
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True, 
	num_workers=NUM_WORKERS,
	pin_memory=True, # acccelerate the transfer cpu to gpu
	persistent_workers=True) # conserve the workers into epochs

val_loader =  torch.utils.data.DataLoader(
	val_dataset,
	batch_size = BATCH_SIZE,
	shuffle = False,
	num_workers = NUM_WORKERS,
	pin_memory = True,
	persistent_workers = True)

print (f"Training Dataset: {len(train_dataset):,} images")
print (f"Validation Dataset: {len(val_dataset):,} images")
print (f"Number of Class: {len(train_dataset.classes)}")

#=======================
# MODEL CREATION
#=======================

print ("Creating the Model resNet50..")

#Creating without pre-trained weight
model = models.resnet50(weights=None)

#sending model to GPU
model = model.cuda()

#Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f"total Parameters: {total_params:,}")
print (f"trained Parameters: {trainable_params:,}")
print (f"Model on the gpu : {next(model.parameters()).is_cuda}")


#=================
# LOSS FUNCTION
#=================

# Cross entropy
criterion = nn.CrossEntropyLoss()

# SGD with momentum
optimizer = optim.SGD(
	model.parameters(),
	lr = LEARNING_RATE,
	momentum = MOMENTUM,
	weight_decay = WEIGHT_DECAY)

print (f"Loss : {criterion}")
print (f"Optimizer : SGD (lr = {LEARNING_RATE}, momentum = {MOMENTUM}, weight_decay = {WEIGHT_DECAY})")


#====================================
# Training for one Epoch
#====================================

def train_one_epoch (model, loader, criterion, optimizer, epoch, device):
	model.train() 

	running_loss = 0.0
	running_corrects = 0
	total_samples = 0
	start_time = time.time()

	for batch_idx, (images, labels) in enumerate(loader):
		# send data to gpu
		images = images.cuda()
		labels = labels.cuda()

		# reset gradient to 0
		optimizer.zero_grad()

		# Forward pass
		outputs = model(images)

		# loss
		loss = criterion(outputs, labels)

		#Backward pass
		loss.backward()

		#weight updating
		optimizer.step()


		#statistics
		_, predicted = torch.max(outputs, 1)
		running_loss += loss.item() * images.size(0)
		running_corrects += (predicted == labels).sum().item()
		total_samples += images.size(0)

		# Show progression
		if (batch_idx +1) % LOG_INTERVAL == 0:
			current_loss = running_loss / total_samples
			current_acc = 100.0 * running_corrects / total_samples
			elapsed = time.time() - start_time
			print (f"Epoch [{epoch + 1}] Batch [{batch_idx+1}/{len(loader)}] "
				f"Loss : {current_loss:.4f} Acc : {current_acc:.2f}% "
				f"Time: {elapsed:.1f}s")
	#stats of the end of epoch
	epoch_loss = running_loss / total_samples
	epoch_acc = 100.0 * running_corrects / total_samples
	epoch_time = time.time() - start_time

	print (f"Epoch [{epoch+1}] ended - Loss: {epoch_loss:.4f} "
		f"Acc: {epoch_acc:.2f}% - Time: {epoch_time:.1f}s")
	return epoch_loss, epoch_acc


#=======================================
# Validation Function
#=======================================


def validate(model, loader, criterion, device):
	model.eval()

	running_loss = 0.0
	running_corrects = 0
	total_samples = 0

	with torch.no_grad():
		for images, labels in loader:
			images = images.cuda()
			labels = labels.cuda()

			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Statistics
			_, predicted = torch.max(outputs, 1)
			running_loss += loss.item() * images.size(0)
			running_corrects += (predicted == labels).sum().item()
			total_samples += images.size(0)

	val_loss = running_loss / total_samples
	val_acc = 100.0 * running_corrects / total_samples

	print(f"Validation - Loss: {val_loss:.4f} Acc: {val_acc:.4f}%")
	return val_loss, val_acc





#===============================================
# SCHEDULER AND MAIN LOOP
#===============================================

# Scheduler : reduce the learning rate of 10x each 30 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print ("\n" + "="*60)
print ("Begin of the Training")
print ("="*60)
print (f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print (f"GPU: {torch.cuda.get_device_name(0)}")
print (f"Batch size: {BATCH_SIZE}")
print (f"Epochs: {NUM_EPOCHS}")
print (f"Learning rate initial: {LEARNING_RATE}")
print (f"Dataset: {len(train_dataset):,} train / {len(val_dataset):,} val")
print ("="*60 + "\n")

best_val_acc = 0.0
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
	current_lr = scheduler.get_last_lr()[0]
	print (f"\n--- Epoch {epoch+1} / {NUM_EPOCHS} (LR: {current_lr:.6f}) ---")

	#Training
	train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, "cuda")

	# Validation
	val_loss, val_acc = validate(model, val_loader, criterion, "cuda")

	# Scheduler step
	scheduler.step()

	# checkpoint saving if it's the best fit model
	if val_acc > best_val_acc:
		best_val_acc = val_acc
		checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
		torch.save({
			'epoch': epoch + 1,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'val_acc': val_acc,
			'val_loss': val_loss,
			'train_acc': train_acc,
			'train_loss': train_loss,
		}, checkpoint_path)
		print (f" best model: {checkpoint_path} (Acc: {val_acc:.2f}%)")

	# Save periodic checkpoint
	if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
		periodic_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
		torch.save({
			'epoch': epoch + 1,
			'model_state_dict': model.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'val_acc': val_acc,
		}, periodic_path)
		print (f" periodic Checkpoint: {periodic_path}")

# End of Training

total_time = time.time() - train_start_time
hours = total_time // 3600
minutes = (total_time % 3600) // 60
print ("\n" + "="*60)
print (f"Training ends in {int(hours)}h {int(minutes)}min")
print (f"Best Accuracy of Validation: {best_val_acc:.2f}%")
print ("="*60)
