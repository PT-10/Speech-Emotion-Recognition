import io
import torch 
import torch.nn as nn
import torch.nn.functional as F
from model import Features

class SCNNB(nn.Module):
    def __init__(self):
        super(SCNNB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * 50, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 64 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = SCNNB()
PATH = "ravdess_ffn.pth"
model.load_state_dict(torch.load(PATH))
model.eval()

def predict(input):
    input = Features.augment_input(input)
    input = Features.extract_features(input)

    # Convert the features to a tensor and unsqueeze to add a batch dimension
    input_tensor = torch.tensor(input).unsqueeze(0)

    # Make a prediction using the model
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.float())
        _, predicted = torch.max(output.data, 1)
        class_label = predicted.item()
    print(f"Class label: {class_label}")
    return class_label

