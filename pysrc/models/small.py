from torch import nn
from torch.nn import functional as F


class SmallFeatureExtractor(nn.Module):
    kernel_size = 5

    def __init__(self, nchannelsin):
        super().__init__()

        # Net architecture
        self.conv1 = nn.Conv2d(in_channels=nchannelsin,
                               out_channels=32,
                               kernel_size=self.kernel_size,
                               # THose are the default values
                               stride=1,
                               padding=0
                               )
        # THe same can be used as max pool 1 and 2
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                     stride=2,
                                     # default value
                                     padding=0)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=48,
                               kernel_size=self.kernel_size)

        # Initialization
        nn.init.normal_(self.conv1.weight, 0, 0.02)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, 0, 0.02)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, input_):
        mb_size = input_.size()[0]
        input_ = self.maxpool(F.relu_(self.conv1(input_)))
        input_ = self.maxpool(F.relu_(self.conv2(input_)))

        return input_.view(mb_size, -1)


class SmallLabelPredictor(nn.Module):
    dim = 100

    def __init__(self, ndimsin, nclasses):
        super().__init__()

        self.fc1 = nn.Linear(ndimsin, self.dim)
        self.fc2 = nn.Linear(self.dim, self.dim)
        self.fc3 = nn.Linear(self.dim, nclasses)

        # Initialization
        self.initialize_lin(self.fc1)
        self.initialize_lin(self.fc2)
        self.initialize_lin(self.fc3, bias=1/nclasses)

    @staticmethod
    def initialize(layer, bias=0):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    def forward(self, input_):
        input_ = F.relu_(self.fc1(input_))
        input_ = F.relu_(self.fc2(input_))
        return self.fc3(input_)


class SmallDomainBin(nn.Module):
    dim = 100

    def __init__(self, ndimsin, domain_lambda):
        super().__init__()

        if domain_lambda:
            # TODO add this
            self.reversal = None
        self.fc1 = nn.Linear(ndimsin, self.dim)
        self.fc2 = nn.Linear(self.dim, 1)

        self.initialize(self.fc1)
        self.initialize(self.fc2)

    @staticmethod
    def initialize(layer):
        nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input_):
        input_ = F.relu_(self.fc1(input_))
        input_ = F.sigmoid(self.fc2(input_))

        return input_
