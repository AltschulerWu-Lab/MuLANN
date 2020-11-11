from torch import nn, sigmoid_
from torch.nn import functional as F
from pysrc.utils.gradient_reversal import GradientReversalLayer


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

    @property
    def ndimsout(self):
        return 1200

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
    def initialize_lin(layer, bias=0):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    def forward(self, input_):
        input_ = F.relu_(self.fc1(input_))
        input_ = F.relu_(self.fc2(input_))
        return self.fc3(input_)


class SmallDomainDiscriminator(nn.Module):
    dim = 100

    def __init__(self, ndimsin, domain_lambda=None, info_zeta=None):
        super().__init__()

        if domain_lambda:
            self.gradient_trsform = GradientReversalLayer(lambda_=domain_lambda, sign=-1)
        elif info_zeta:
            self.gradient_trsform = GradientReversalLayer(lambda_=info_zeta, sign=1)
        else:
            print('No gradient reversal nor information passing')
            self.gradient_trsform = None
        self.fc1 = nn.Linear(ndimsin, self.dim)
        self.fc2 = nn.Linear(self.dim, 1)

        self.initialize(self.fc1)
        self.initialize(self.fc2)

    @staticmethod
    def initialize(layer):
        nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input_):
        if self.gradient_trsform:
            input_ = self.gradient_trsform(input_)
        input_ = F.relu_(self.fc1(input_))
        input_ = sigmoid_(self.fc2(input_))

        return input_


class SmallNet(nn.Module):
    def __init__(self, nchanelsin, nclasses, domain_lambda, info_zeta):
        super().__init__()

        self.feature_extractor = SmallFeatureExtractor(nchannelsin=nchanelsin)
        self.label_predictor = SmallLabelPredictor(ndimsin=self.feature_extractor.ndimsout,
                                                   nclasses=nclasses)
        if domain_lambda:
            self.domain_discriminator = SmallDomainDiscriminator(ndimsin=self.feature_extractor.ndimsout,
                                                             domain_lambda=domain_lambda)
        else:
            self.domain_discriminator = None

        if info_zeta:
            self.info_predictor = SmallDomainDiscriminator(ndimsin=self.feature_extractor.ndimsout,
                                                           info_zeta=info_zeta)
        else:
            self.info_predictor = None

    def forward(self, input_):
        input_ = self.feature_extractor.forward(input_)
        class_predictions = self.label_predictor.forward(input_)
        domain_predictions = self.domain_discriminator(input_) if self.domain_discriminator else None
        info_predictions = self.info_predictor(input_) if self.info_predictor else None

        return class_predictions, domain_predictions, info_predictions
