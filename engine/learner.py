import os
import torch
from .data_loader import DataLoader
from .trainer import Trainer
from .plots import plot_history
from .visualize import ShowData, Classified


class Learner(object):
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = Trainer(model, train_loader, test_loader, loss_fn, optimizer, scheduler)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classified = None
        
    def show_images(self,num=5):
        ShowData.random_images(self.train_loader, num=num)

    def run_epochs(self, epochs):
        self.trainer.run(epochs)

    def show_history(self):
        plot_history(self.trainer.get_train_history(), self.trainer.get_test_history())

    def plot_misclassified(self):
        if not self.classified:
            self.classified = Classified(self.model, self.test_loader)
        classified = self.classified
        classified.plot_misclassified(labels=DataLoader.cifar10_classes)
  
    def get_misclassified(self, number=5):
        if not self.classified:
            self.classified = Classified(self.model, self.test_loader)
        classified = self.classified
        misclassified_images, ground_truth, predicted = classified.get_misclassified(number=number)
        return misclassified_images, ground_truth, predicted
    
    def classwise_accuracy(self):
        if not self.classified:
            self.classified = Classified(self.model, self.test_loader)
        classified = self.classified
        classified.classwise_accuracy()
