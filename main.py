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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bnorm = nn.BatchNorm2d(
            out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv(self.relu(self.bnorm(x)))


# Define model
class Main(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet161 = nn.Sequential(*list(DenseNet161.children())[:-1])
        # first bottleneck layer
        self.btl1 = Bottleneck(
            in_channels=2208, out_channels=512, kernel_size=1, stride=1, padding=1
        )

    def forward(self, x):
        x = self.densenet161(x)
        # x = self.btl1(x)
        return x


model = Main().to(device)

# print(model.densenet161.features.denseblock4.denselayer24.conv2.weight.size())
# print(model)
# for name, param in model.named_parameters():
#     print(name, param.size())
# print(model.densenet161.features.norm5.weight.size())


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


epochs = 20
steps = 0
print_every = 20
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(train_loader):
        inputs, targets = images, labels
        steps += 1

        # if torch.cuda.is_available():
        #     inputs, targets = inputs.cuda(), labels.cuda()
        inputs = inputs.to(device)
        targets = labels.to(device)

        print("----------------------------------", inputs.size())
        print("----------------------------------", targets.size())

        optimizer.zero_grad()

        output = model.forward(inputs)

        targets = torch.argmax(targets, dim=0)
        output = torch.argmax(output, dim=0)

        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            accuracy = 0
            valid_loss = 0
            for ii, (images, labels) in enumerate(test_loader):
                inputs, labels = images, labels
                with torch.no_grad():
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = model.forward(inputs)
                    valid_loss += loss_fn(output, labels).item()
                    ps = torch.exp(output).data
                    equality = labels.data == ps.max(1)[1]
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            print(
                "Epoch: {}/{}.. ".format(e + 1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                "Test Loss: {:.3f}.. ".format(valid_loss / len(test_loader)),
                "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)),
            )

            running_loss = 0
            model.train()


# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         # loss = loss_fn(pred, y)

#         # # Backpropagation
#         # optimizer.zero_grad()
#         # loss.backward()
#         # optimizer.step()

#         # if batch % 100 == 0:
#         #     loss, current = loss.item(), batch * len(X)
#         #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_loader, model, loss_fn, optimizer)
# print("Done!")
