import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, TensorDataset

# Paths and constants
IMAGE_DIRS = ["antrenare/dad/", "antrenare/deedee/", "antrenare/dexter/", "antrenare/mom/"]
ANNOTATION_FILES = ["antrenare/dad_annotations.txt", "antrenare/deedee_annotations.txt", "antrenare/dexter_annotations.txt", "antrenare/mom_annotations.txt"]
PNet_INPUT_SIZE = 12
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Load annotations
def load_annotations(annotation_files, image_dirs):
    for annotation_file, image_dir in zip(annotation_files, image_dirs):
        with open(annotation_file, "r") as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split()
            image_path = os.path.join(image_dir, parts[0])
            bbox = list(map(int, parts[1:5]))
            annotations.append((image_path, bbox))
    return annotations

def iou(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    intersection_over_union = inter_area / float(box_a_area + box_b_area - inter_area)

    return intersection_over_union

# Preprocess images and generate training data
def preprocess_images(annotations):
    positive_samples = []
    negative_samples = []
    bbox_targets = []
    labels = []

    for image_path, bbox in annotations:
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Rescale the image to 12x12
        img_rescaled = cv2.resize(img, (PNet_INPUT_SIZE, PNet_INPUT_SIZE))

        # Positive sample
        x_min, y_min, x_max, y_max = bbox[:4]  # Use only the first bounding box

        positive_samples.append(img_rescaled)
        bbox_targets.append([x_min, y_min, x_max, y_max])
        labels.append(1)  # Label for face

        # Generate random negative samples
        for _ in range(4):  # Generate 4 negatives per image
            # Randomly sample a bounding box
            x_min = np.random.randint(0, PNet_INPUT_SIZE - 1)
            y_min = np.random.randint(0, PNet_INPUT_SIZE - 1)
            x_max = np.random.randint(x_min + 1, PNet_INPUT_SIZE)
            y_max = np.random.randint(y_min + 1, PNet_INPUT_SIZE)

            # Calculate IoU
            iou_val = iou(bbox, [x_min, y_min, x_max, y_max])

            # If IoU is less than 0.3, consider it as a negative sample
            while iou_val > 0.3:
                x_min = np.random.randint(0, PNet_INPUT_SIZE - 1)
                y_min = np.random.randint(0, PNet_INPUT_SIZE - 1)
                x_max = np.random.randint(x_min + 1, PNet_INPUT_SIZE)
                y_max = np.random.randint(y_min + 1, PNet_INPUT_SIZE)
                iou_val = iou(bbox, [x_min, y_min, x_max, y_max])

            # Negative sample
            negative_samples.append(img_rescaled)
            bbox_targets.append([x_min, y_min, x_max, y_max])
            labels.append(0)  # Label for non-face

    # Combine positive and negative samples
    X = np.concatenate([positive_samples, negative_samples], axis=0)
    y = np.array(labels, dtype=np.float32)

    # Prepare bounding boxes and pad with zeros to ensure shape (BATCH_SIZE, 4)
    padded_bbox_targets = np.zeros((len(X), 4), dtype=np.float32)
    for i, bbox in enumerate(bbox_targets):
        padded_bbox_targets[i, :] = bbox

    return torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(padded_bbox_targets, dtype=torch.float32)

# Define PNet model
class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1) # 2d convolutional layer with 3 input channels for RGB, 10 output channels, 3x3 kernel
        self.prelu1 = nn.PReLU(10) # activation function
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # max pooling layer with 2x2 kernel, stride 2, output size is rounded up
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1) # 2d convolutional layer with 10 input channels, 16 output channels, 3x3 kernel
        self.prelu2 = nn.PReLU(16) # activation function
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1) # 2d convolutional layer with 16 input channels, 32 output channels, 3x3 kernel
        self.prelu3 = nn.PReLU(32) # activation function
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)  # Change output channels to 2 for classification
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)  # Keep output channels as 4 for bbox regression

    def forward(self, x):
        x = self.prelu1(self.conv1(x)) # Apply convolution and activation
        x = self.pool1(x) # Apply max pooling
        x = self.prelu2(self.conv2(x)) # Apply convolution and activation
        x = self.prelu3(self.conv3(x)) # Apply convolution and activation
        face_cls = torch.sigmoid(self.conv4_1(x)) # Apply convolution and sigmoid activation for classification
        bbox_reg = self.conv4_2(x) # Apply convolution for bbox regression
        return face_cls, bbox_reg

# Training loop
def train_pnet(model, dataloader, bbox_targets, epochs=10, lr=0.001):
    model.train()
    cls_loss_fn = nn.BCELoss()  # Classification loss (binary cross-entropy)
    bbox_loss_fn = nn.MSELoss()  # Bounding box regression loss (mean squared error)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_cls_loss, total_bbox_loss = 0, 0
        for i, (X_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad() # clear gradients before each iteration
            face_cls, bbox_reg = model(X_batch)

            # Ensure target and predictions have the same shape for classification
            cls_loss = cls_loss_fn(face_cls.view(-1), y_batch.repeat_interleave(2).view(-1))
            bbox_loss = bbox_loss_fn(bbox_reg.view(bbox_reg.size(0), -1), bbox_targets[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            total_loss = cls_loss + bbox_loss

            # Backpropagation
            total_loss.backward() # parameter importance regarding the loss (higher gradient, higher importance, bigger loss)
            optimizer.step() # update weights based on gradients and learning rate

            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Classification Loss: {total_cls_loss:.4f}, BBox Loss: {total_bbox_loss:.4f}")

# Load data and annotations
annotations = load_annotations(ANNOTATION_FILES, IMAGE_DIRS)
X, y, bbox_targets = preprocess_images(annotations)

# Create dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize and train PNet
pnet = PNet()
train_pnet(pnet, dataloader, bbox_targets, epochs=EPOCHS, lr=LEARNING_RATE)

# Save the trained PNet
torch.save(pnet.state_dict(), "trained_pnet.pth")