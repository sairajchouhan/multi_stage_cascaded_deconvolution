import torch
from torch import nn
from nyu_v2_dataset import NYUv2Dataset
import os


debug = bool(os.getenv("DEBUG", default=False))


DenseNet161 = torch.hub.load("pytorch/vision:v0.10.0", "densenet161", pretrained=True)

batch_size = 10

device = (
    torch.device("mps")
    if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    else torch.device("cpu")
)


# device = torch.device("cpu")


print("Using device:", device.type.upper())


train_dataset = NYUv2Dataset(
    train=True,
)
test_dataset = NYUv2Dataset(
    train=False,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", next(iter(train_loader)))
print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", next(iter(train_loader)))


# make a nice class for bottleneck layer
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.bnorm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.bnorm = nn.BatchNorm2d(in_channels)
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
            in_channels=2208, out_channels=512, kernel_size=1, stride=1, padding=0
        )

        # 1st deconvolution layer
        self.deconv1 = DeConv(
            in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=1
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=1, padding=0)

        # 2nd deconvolution layer
        self.deconv2 = DeConv(
            in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=1
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)

        # 3rd deconvolution layer
        self.deconv3 = DeConv(
            in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1
        )
        self.avgpool3 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)

        # 4th deconvolution layer
        self.deconv4 = DeConv(
            in_channels=128, out_channels=1, kernel_size=5, stride=2, padding=1
        )
        self.avgpool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)

    def forward(self, x):
        x = self.densenet161(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after densenet161", x.size())

        x = self.btl1(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after btl1", x.size())

        x = self.deconv1(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after deconv1", x.size())

        x = self.avgpool1(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after avgpool1", x.size())

        x = self.deconv2(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after deconv2", x.size())

        x = self.avgpool2(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after avgpool2", x.size())

        x = self.deconv3(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after deconv3", x.size())

        x = self.avgpool3(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after avgpool3", x.size())

        x = self.deconv4(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after deconv4", x.size())

        x = self.avgpool4(x)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ after avgpool4", x.size())

        return x


model = Main().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        if debug:
            print(
                "-------------------------------TRAIN------------------------------------"
            )
            print("X ", X.size())
            print("y ", y.size())
            print("pred ", pred.size())
            print(
                "-------------------------------TRAIN END------------------------------------"
            )

        # convert to 3d tensor
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = 5


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if debug:
                print(
                    "\n-------------------------------TEST------------------------------------"
                )
                print("X ", X.size())
                print("y ", y.size())
                print("pred ", pred.size())
                print(
                    "-------------------------------TEST END------------------------------------\n"
                )
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test set: Avg loss: {test_loss:>8f} \n")


# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_loader, model, loss_fn, optimizer)
#     test(test_loader, model, loss_fn)

print("Done!")
