import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim


class Extractor(nn.Module):
    def __init__(self, input_dim=310, hidden_dim=500, output_dim=300):
        super(Extractor, self).__init__()
        self.class_feasure = nn.Sequential()
        self.class_feasure.add_module('e_fc1', nn.Linear(input_dim, hidden_dim))
        self.class_feasure.add_module('e_bn1', nn.BatchNorm1d(hidden_dim))
        self.class_feasure.add_module('e_sigmoid1', nn.Sigmoid())
        # self.class_feasure.add_module('e_drop1', nn.Dropout2d())
        # self.class_feasure.add_module('e_fc2', nn.Linear(hidden_dim, hidden_dim))
        # self.class_feasure.add_module('e_bn2', nn.BatchNorm1d(hidden_dim))
        # self.class_feasure.add_module('e_relu1', nn.ReLU(True))
        self.class_feasure.add_module('e_fc3', nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.class_feasure(x)


class Classifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=500, output_dim=4):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(input_dim, hidden_dim))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(hidden_dim))
        self.class_classifier.add_module('c_sigmoid1', nn.Sigmoid())
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(hidden_dim, hidden_dim))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(hidden_dim))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.class_classifier(x)
        return F.softmax(x, dim=1)


class Discriminator(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=300):
        super(Discriminator, self).__init__()
        self.class_discriminator = nn.Sequential()
        self.class_discriminator.add_module('d_fc1', nn.Linear(input_dim, hidden_dim))
        self.class_discriminator.add_module('d_bn1', nn.BatchNorm1d(hidden_dim))
        self.class_discriminator.add_module('d_sigmoid1', nn.Sigmoid())
        # self.class_discriminator.add_module('d_drop1', nn.Dropout2d())
        # self.class_discriminator.add_module('d_fc2', nn.Linear(hidden_dim, hidden_dim))
        # self.class_discriminator.add_module('d_bn2', nn.BatchNorm1d(hidden_dim))
        # self.class_discriminator.add_module('d_relu1', nn.ReLU(True))
        self.class_discriminator.add_module('d_fc3', nn.Linear(hidden_dim, 1))
        self.class_discriminator.add_module('d_sigmoid2', nn.Sigmoid())

    def forward(self, x):
        return self.class_discriminator(x)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = Extractor()
        self.classifier = Classifier()
        self.domain_classifier = Discriminator()

    def forward(self, input_data, alpha=1, source=True):
        feature = self.feature(input_data)
        class_output = self.classifier(feature)
        domain_output = self.get_adversarial_result(feature, source, alpha)
        return class_output, domain_output

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def get_name(self):
        return "DANN"


class ZeroLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha=0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ADDA(nn.Module):
    def __init__(self, device, data, label):
        super(ADDA, self).__init__()
        self.device = device
        self.src_feature = Extractor()
        self.tar_feature = Extractor()
        self.classifier = Classifier()
        self.domain_classifier = Discriminator()
        self.src_train(data, label)

    def src_train(self, data, label):
        src_model = torch.nn.Sequential(self.src_feature, self.classifier)
        nepoch = 500
        loss_class = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(src_model.parameters(), lr=0.01)
        for epoch in range(nepoch):
            output = src_model(data)
            loss = loss_class(output, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def forward(self, input_data, alpha=1, source=True):
        if source:
            feature = self.src_feature(input_data)
        else:
            feature = self.tar_feature(input_data)
        class_output = self.classifier(feature)
        domain_output = self.get_adversarial_result(feature, source, alpha)
        return class_output, domain_output

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
            x = ZeroLayerF.apply(x)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
            x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def get_name(self):
        return "ADDA"
