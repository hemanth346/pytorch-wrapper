import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from .data_loader import DataLoader
from .utils import UnNormalize, unnormalize

# class UnNormalize(object):
#     # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/11
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return t

class ShowData(object):
    def __init__(self):
        pass
    @staticmethod
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    @staticmethod
    def random_images(data_loader, classes = DataLoader.cifar10_classes, num=5):
        # get some random training images
        dataiter = iter(data_loader)
        images, labels = dataiter.next()
        # img_list = random.sample(range(1, images.shape[0]), num)
        img_list = range(1,num+1)
        # show images
        # print('shape:', images.shape)
        # print('Input images to model')
        ShowData.imshow(torchvision.utils.make_grid(images[img_list], nrow=5, padding=2))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in img_list))

class Classified(object):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        pass

    def get_misclassified(self, number=20):
        # predict_generator
        '''
            Generates output predictions for the input samples.
            predict(x, batch_size=None, verbose=0, steps=None, callbacks=None,
            max_queue_size=10, workers=1, use_multiprocessing=False)
        '''
        if self.data_loader:
            data_loader = self.data_loader

        dataiter = iter(data_loader)
        data, labels = dataiter.next()
        misclassified_images = torch.rand(number,*data.shape[-3:]) * 0
        ground_truth = torch.rand(number,1)*0
        predicted = torch.rand(number,1)*0
        false_positives = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                false_picker = torch.flatten(pred)-target
                index = 0
                for val in false_picker:
                    if (val != 0):
                        misclassified_images[false_positives] = data[index, :, :, :]
                        ground_truth[false_positives] = target[index]
                        predicted[false_positives] = pred[index]
                        false_positives += 1
                        if (false_positives >= number):
                            break

                    index += 1

                if (false_positives >= number):
                        break
        return misclassified_images, ground_truth, predicted

    def plot_misclassified(self, labels = [], number=20, title='', mean=DataLoader.cifar10_mean, std=DataLoader.cifar10_std):
        if self.data_loader:
            data_loader = self.data_loader

        misclassified_images, ground_truth, predicted = self.get_misclassified(number)
        cols = 4
        rows = (number//cols) + (number % cols)
        fig = plt.figure()
        fig.set_figheight(rows*3)
        fig.set_figwidth(cols*3)
        fig.suptitle(title) #super title
        unorm = 0
        if mean and std:
            # unorm = UnNormalize(mean=mean, std=std)
            unorm = 1

        for i in range((rows*cols)):
            plt.subplot(rows,cols,i+1)
            # plt.tight_layout()
            # transposing as the channels are first here and plt expects them to be last
            img = misclassified_images[i, :, :,:]
            if unorm:
                # img = unorm(misclassified_images[i, :, :,:])*155.
                img = unnormalize(img)
            else:
                raise ValueError('Provide Mean and Standard Deviation for displaying correct image')
            img = np.transpose(img.numpy(), (1, 2, 0))
            plt.imshow(img.astype('uint8'), interpolation='none')
            if labels:
                plt.title("GT:{}, Pred:{}".format(labels[int(ground_truth[i])], labels[int(predicted[i])]))
            else:
                plt.title("GT:{}, Pred:{}".format(ground_truth[i].numpy(), predicted[i].numpy()))
            plt.xticks([])
            plt.yticks([])

    def classwise_accuracy(self, classes = DataLoader.cifar10_classes):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print(f'Accuracy of {classes[i]:<10} : {(100 * class_correct[i] / class_total[i]):.2f}%')
