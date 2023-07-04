import torch.nn as nn

from gaze_estimation.resnet import resnet50


class gaze_network(nn.Module):
    def __init__(self):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 4),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.gaze_fc(feature)
        gaze = feature[:, :2]
        head = feature[:, 2:]

        return gaze, head
