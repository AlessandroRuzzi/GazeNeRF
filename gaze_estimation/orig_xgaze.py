import torch.nn as nn

from gaze_estimation.resnet import resnet50

class gaze_network_orig(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network_orig, self).__init__()
        self.gaze_network = resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze