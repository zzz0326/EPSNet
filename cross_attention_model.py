import torch
import torch.nn as nn


def conv2d(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)


def deconv2d(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)


def relu(inplace=True):  # Change to True?
    return nn.ReLU(inplace)


def maxpool2d():
    return nn.MaxPool2d(2)


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_deconv_layers(cfg):
    layers = []
    in_channels = 512
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}


def erp_encoder():
    return make_conv_layers(cfg['E'])


def cmp_encoder():
    return make_conv_layers(cfg['C'])



class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.erp_encoder = erp_encoder()
        self.cmp_encoder = cmp_encoder()
        # self.cmp_FC = nn.Linear(in_features=512 * 2 * 2, out_features=512)

        self.erp_Linear = nn.Linear(in_features=20 * 10 * 512, out_features=512)
        self.cmp_Linear = nn.Linear(in_features=5 * 5 * 512, out_features=512)

        self.q_linear = nn.Linear(512, 512)
        self.v_linear = nn.Linear(512, 512)
        self.k_linear = nn.Linear(512, 512)

        self.relu = nn.ReLU(inplace=True)
        self.regression = nn.Linear(512, 1)

        self.Max = nn.MaxPool2d(2)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1024 + 128, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 6)
            # 六个面+contrast
        )

    def forward(self, erp, cmp1, cmp2, cmp3, cmp4, cmp5, cmp6):
        erp = self.erp_encoder(erp)
        cmp1 = self.cmp_encoder(cmp1)
        cmp2 = self.cmp_encoder(cmp2)
        cmp3 = self.cmp_encoder(cmp3)
        cmp4 = self.cmp_encoder(cmp4)
        cmp5 = self.cmp_encoder(cmp5)
        cmp6 = self.cmp_encoder(cmp6)

        erp = erp.view(-1, 20 * 10 * 512)
        cmp1 = cmp1.view(-1, 5 * 5 * 512)
        cmp2 = cmp2.view(-1, 5 * 5 * 512)
        cmp3 = cmp3.view(-1, 5 * 5 * 512)
        cmp4 = cmp4.view(-1, 5 * 5 * 512)
        cmp5 = cmp5.view(-1, 5 * 5 * 512)
        cmp6 = cmp6.view(-1, 5 * 5 * 512)

        erp = self.erp_Linear(erp)
        cmp1 = self.cmp_Linear(cmp1)
        cmp2 = self.cmp_Linear(cmp2)
        cmp3 = self.cmp_Linear(cmp3)
        cmp4 = self.cmp_Linear(cmp4)
        cmp5 = self.cmp_Linear(cmp5)
        cmp6 = self.cmp_Linear(cmp6)

        q = self.q_linear(erp).unsqueeze(1)

        k1 = self.k_linear(cmp1).unsqueeze(2)
        k2 = self.k_linear(cmp2).unsqueeze(2)
        k3 = self.k_linear(cmp3).unsqueeze(2)
        k4 = self.k_linear(cmp4).unsqueeze(2)
        k5 = self.k_linear(cmp5).unsqueeze(2)
        k6 = self.k_linear(cmp6).unsqueeze(2)

        v1 = self.v_linear(cmp1).unsqueeze(1)
        v2 = self.v_linear(cmp2).unsqueeze(1)
        v3 = self.v_linear(cmp3).unsqueeze(1)
        v4 = self.v_linear(cmp4).unsqueeze(1)
        v5 = self.v_linear(cmp5).unsqueeze(1)
        v6 = self.v_linear(cmp6).unsqueeze(1)

        a1 = self.relu(torch.matmul(q, k1))
        a2 = self.relu(torch.matmul(q, k2))
        a3 = self.relu(torch.matmul(q, k3))
        a4 = self.relu(torch.matmul(q, k4))
        a5 = self.relu(torch.matmul(q, k5))
        a6 = self.relu(torch.matmul(q, k6))

        o1 = self.relu(torch.matmul(a1, v1)).squeeze(1)
        o2 = self.relu(torch.matmul(a2, v2)).squeeze(1)
        o3 = self.relu(torch.matmul(a3, v3)).squeeze(1)
        o4 = self.relu(torch.matmul(a4, v4)).squeeze(1)
        o5 = self.relu(torch.matmul(a5, v5)).squeeze(1)
        o6 = self.relu(torch.matmul(a6, v6)).squeeze(1)

        out = self.regression(o1)
        out = torch.cat((out, self.regression(o2)), dim=1)
        out = torch.cat((out, self.regression(o3)), dim=1)
        out = torch.cat((out, self.regression(o4)), dim=1)
        out = torch.cat((out, self.regression(o5)), dim=1)
        out = torch.cat((out, self.regression(o6)), dim=1)

        return out

