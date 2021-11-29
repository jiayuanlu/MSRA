from __future__ import print_function
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
CUDA_VISIBLE_DEVICES=0
torch.cuda.empty_cache()

class My_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,weight):
        ctx.save_for_backward(input, weight)
        output=input.mm(weight.T)
        return output
    
    @staticmethod
    def backward(ctx,grad_output):
        input,weight=ctx.saved_tensors
        grad_input=grad_weight=None
        if ctx.needs_input_grad[0]:
            grad_input=grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight=(grad_output.T).matmul(input)
        return grad_input,grad_weight


class My_linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True)->None:
        super(My_linear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # nn.Parameter是特殊Variable
        self.bias = nn.Parameter(torch.randn(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-bound, bound)

    def forward(self,input):
        # result=input.mm(self.weight.T)
        # return result
        return My_function.apply(input, self.weight)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32,3,1)
        self.conv2 = nn.Conv2d(32, 64,3,1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.linear1=My_linear(9216,128)
        self.linear2=My_linear(128,10)

    def forward(self,data):
        # x = data.view(-1, 784)
        x = self.conv1(data)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        output=self.linear1(x)
        output=F.relu(output)
        output=self.dropout2(output)
        output=self.linear2(output)
        output=F.log_softmax(output,dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Add profile function
def profile(model, device, train_loader):
    dataiter = iter(train_loader)
    data, target = dataiter.next()
    data, target = data.to(device), target.to(device)
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        model(data[0].reshape(1,1,28,28))
    print(prof)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # profile model
    train_time=0
    test_time=0
    print("Start profiling...")
    profile(model, device, train_loader)
    print("Finished profiling.")
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        torch.cuda.synchronize()
        end = time.time()
        train_time+=(end-start)
        print(end-start)

        torch.cuda.synchronize()
        start1 = time.time()
        test(model, device, test_loader)
        torch.cuda.synchronize()
        end1 = time.time()
        test_time+=(end1-start1)
        print(end1-start1)
        scheduler.step()
    
    train_time/=14
    test_time/=14
    print(train_time)
    print(test_time)

    if args.save_model:
        print("Our model: \n\n", model, '\n')
        print("The state dict keys: \n\n", model.state_dict().keys())
        torch.save(model.state_dict(), "mnist_cnn.pt")
        state_dict = torch.load('mnist_cnn.pt')
        print(state_dict.keys())


if __name__ == '__main__':
    main()
