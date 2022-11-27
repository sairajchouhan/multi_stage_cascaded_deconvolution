import torch
from torch import nn
from nyu_v2_dataset import NYUv2Dataset

DenseNet161 = torch.hub.load("pytorch/vision:v0.10.0", "densenet161", pretrained=True)

batch_size = 12

# device = (
#     torch.device("mps")
#     if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
#     else torch.device("cpu")
# )


device = torch.device("cpu")


print("Using device:", device.type.upper())


train_dataset = NYUv2Dataset(
    train=True,
)
test_dataset = NYUv2Dataset(
    train=False,
)
print("test dataset is =====================================", len(train_dataset))
print("test dataset is =====================================", len(test_dataset))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


# make a nice class for bottleneck layer
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bnorm = nn.BatchNorm2d(
            in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.bnorm = nn.BatchNorm2d(
            in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.deconv(x)
        return x


# Define model
class Main(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet161 = nn.Sequential(*list(DenseNet161.children())[:-1])
        # first bottleneck layer
        self.btl1 = Bottleneck(
            in_channels=2208, out_channels=512, kernel_size=1, stride=1, padding=1
        )

        # 1st deconvolution layer
        self.deconv1 = DeConv(
            in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=1, padding=0)

        # 2nd deconvolution layer
        self.deconv2 = DeConv(
            in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)

        # 3rd deconvolution layer
        self.deconv3 = DeConv(
            in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2
        )
        self.avgpool3 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)

        # 4th deconvolution layer
        self.deconv4 = DeConv(
            in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2
        )
        self.avgpool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)

    def forward(self, x):
        x = self.densenet161(x)
        x = self.btl1(x)
        x = self.deconv1(x)
        x = self.avgpool1(x)
        x = self.deconv2(x)
        x = self.avgpool2(x)
        x = self.deconv3(x)
        x = self.avgpool3(x)
        x = self.deconv4(x)
        x = self.avgpool4(x)

        return x


model = Main().to(device)

# print(model.densenet161.features.denseblock4.denselayer24.conv2.weight.size())
# print(model)
# for name, param in model.named_parameters():
#     print(name, param.size())
# print(model.densenet161.features.norm5.weight.size())


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        # convert to 3d tensor
        loss = loss_fn(pred, y)
        print()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for param in model.parameters():
    print(type(param), param.size())

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(test_loader, model, loss_fn, optimizer)
print("Done!")
