import torchvision.transforms as transforms
import torchvision as tv
import torch as tc
#TODO : main으로부터 batch사이즈 넘겨 받게 구성할 것.
batch_sz= 128
transform = transforms.Compose([transforms.ToTensor()], transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)), transforms.Resize(32))
dataset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataload = tc.utils.data.DataLoader(dataset, )