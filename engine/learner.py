import os
import torch
from .data_loader import DataLoader
from .trainer import Trainer
from .plots import plot_history as display_history
from .visualize import ShowData, Classified


class Learner(object):
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = Trainer(model, train_loader, test_loader, loss_fn, optimizer)

    def show_images(self,num=5):
        ShowData.random_images(self.train_loader, num=num)

    def run_epochs(self, epochs):
        self.trainer.run(epochs)

    def plot_history(self):
        display_history(self.trainer.get_train_history(), self.trainer.get_test_history())

    def plot_misclassified(self):
        classified = Classified(self.model, self.test_loader)
        classified.plot_misclassified(labels=DataLoader.cifar10_classes)
        pass

    def classwise_accuracy(self):
        classified = Classified(self.model, self.test_loader)
        classified.classwise_accuracy()


