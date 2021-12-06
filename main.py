import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from torchvision import transforms

from data import Dataset
from alexnet import AlexNet
from torch.utils.model_zoo import load_url as load_state_dict_from_url

DATA_DIR = '/content/drive/MyDrive/PACS/PACS/'

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

DEVICE = 'cuda'
NUM_CLASSES = 7
BATCH_SIZE = 256
LR = 1e-3            # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
NUM_EPOCHS = 30      # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20       # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10

# Define transforms for training phase
train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define transforms for the evaluation phase
eval_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Prepare Pytorch train/test Datasets
source_dataset = Dataset(DATA_DIR, dataset='photo',  transform=train_transform)
target_dataset = Dataset(DATA_DIR, dataset='art_painting', transform=eval_transform)

# Check dataset sizes
print('Source Dataset: {}'.format(len(source_dataset)))
print('Target Dataset: {}'.format(len(target_dataset)))

source_dataloader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

net = AlexNet() # Loading AlexNet model
state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=True)
net.load_state_dict(state_dict,strict=False)

net.classifier[6] = nn.Linear(4096, NUM_CLASSES)

criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy
parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

current_step = 0

for epoch in range(NUM_EPOCHS):

    # Iterate over the dataset
    for images, labels in source_dataloader:

      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net.train() # Sets module in training mode
      optimizer.zero_grad() # Zero-ing the gradients
      outputs = net(images) # Forward pass to the network
      loss = criterion(outputs, labels) # Compute loss based on output and ground truth

      if current_step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(current_step, loss.item()))

      loss.backward()  # backward pass: computes gradients
      optimizer.step() # update weights based on accumulated gradients

      current_step += 1

    # Step the scheduler
    scheduler.step()

net.train(False) # Set Network to evaluation mode

running_corrects = 0
for images, labels in target_dataloader:

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = net(images)
    _, preds = torch.max(outputs.data, 1)
    running_corrects += torch.sum(preds == labels.data).data.item()

accuracy = running_corrects / float(len(target_dataset))

print('Accuracy on the target domain: {}'.format(accuracy))