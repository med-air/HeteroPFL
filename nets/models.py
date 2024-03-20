import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch.nn as nn
import torch.nn.functional as F


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, bn_affine=True, **kwargs):
        super(DigitModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64, affine=bn_affine)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64, affine=bn_affine)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128, affine=bn_affine)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048, affine=bn_affine)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512, affine=bn_affine)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x, feature=False, activation=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        if activation:
            last_act = x
        feat = F.relu(x)

        x = self.fc3(feat)
        if feature and activation:
            return last_act, feat, x
        elif feature and not activation:
            return feat, x
        elif activation and not feature:
            return last_act, x
        else:
            return x


