"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import Dataset


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        # femnist images are grayscale (1 channel) and 28x28
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # After two conv(5) + two pool(2) on 28x28 -> final feature map is 16 x 4 x 4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)  # femnist has multiple character classes; keep output flexible

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

# femnist: grayscale images -> single-channel normalization
pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


def collate_fn(batch):
    """Custom collate for HuggingFace-style examples containing PIL images.

    Converts each example's 'image' (PIL) to a tensor using pytorch_transforms
    and stacks them into a batched tensor; converts 'character' to a long
    tensor of labels.
    """
    imgs = [pytorch_transforms(x["image"]) for x in batch]
    labels = torch.tensor([int(x["character"]) for x in batch], dtype=torch.long)
    return {"img": torch.stack(imgs), "label": labels}


def load_data(partition_id: int, num_partitions: int, batch_size: int, class_filter=None, is_malicious=False):
    """Load partition FEMNIST data.

    Parameters:
    - partition_id: which federated partition to load
    - num_partitions: total number of partitions used by the partitioner
    - batch_size: DataLoader batch size
    - class_filter: optional list of character ids to keep (if provided, filters both train and test)

    Returns: trainloader, testloader
    """
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)

    # The dataset only provides 'train', so create a local train/test split
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Optionally filter to a subset of classes
    if class_filter is not None:
        def keep_example(example):
            return example["character"] in class_filter

        partition_train_test["train"] = partition_train_test["train"].filter(keep_example)
        partition_train_test["test"] = partition_train_test["test"].filter(keep_example)

    # Construct dataloaders. The underlying examples may contain PIL images
    # (the 'image' column), so use a custom collate_fn that applies the
    # pytorch_transforms and stacks tensors into batches.
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    # print size of train and test sets
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size, collate_fn=collate_fn)

    if is_malicious:
        trainloader = make_malicious_trainloader(trainloader, num_classes=62, shift=1)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader.

    femnist only exposes a 'train' split. We split it locally to create a test set
    that approximates a held-out evaluation partition (20%).
    """
    # Load entire train set and split to create a test portion
    train_dataset = load_dataset("flwrlabs/femnist", split="train")
    split = train_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split["test"]
    # Use the same collate_fn so PIL images are converted when batching
    return DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

class ShiftLabelDataset(Dataset):
    def __init__(self, base_dataset, num_classes, shift=1):
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.shift = shift

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        example = self.base_dataset[idx]

        # HuggingFace Dataset は dict を返す
        new_example = dict(example)
        new_example["character"] = (
                                           example["character"] + self.shift
                                   ) % self.num_classes

        return new_example

def make_malicious_trainloader(trainloader, num_classes=62, shift=1):
    return DataLoader(
        ShiftLabelDataset(
            trainloader.dataset,
            num_classes=num_classes,
            shift=shift,
        ),
        batch_size=trainloader.batch_size,
        shuffle=True,
        num_workers=trainloader.num_workers,
        collate_fn=trainloader.collate_fn,  # ★重要
        pin_memory=True,
    )