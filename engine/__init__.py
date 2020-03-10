import os 
import torch
from .version import __version__
# CUDA?
cuda = torch.cuda.is_available()
os.environ['device'] = torch.device("cuda" if cuda else "cpu")


# from utils import UnNormalize#, validate_loss
# from trainer import Trainer
# from learner import Learner
# from plots import plot_history
# from visualize import ShowData, Classified
# from data_loader import DataLoader
# from models import Net, BasicBlock, Bottleneck, ResNet