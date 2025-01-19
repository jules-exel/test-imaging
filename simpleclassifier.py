import timm
from torch import nn


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)