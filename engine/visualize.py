import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from .data_loader import CIFAR10Data
from .utils import UnNormalize, unnormalize_chw_image, unnormalize_hwc_image

class ShowData(object):
    def __init__(self):
        pass
    @staticmethod
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    @staticmethod
    def show_images(data_loader, num=5, mean=CIFAR10Data.mean, std=CIFAR10Data.std, classes = CIFAR10Data.classes):
        # get some random training images
        images, labels = next(iter(data_loader))
        # # change from BCHW to BHWC
        # images = images.permute(0, 2, 3, 1)
        img_list = random.sample(range(1, images.shape[0]), num)
        for idx in img_list:
            img = unnormalize_chw_image(images[idx], mean, std)
            plt.imshow(img, interpolation='bilinear')
            plt.title(classes[labels[idx]])
        plt.imshow()


class Validator(object):
    """
        Validate given model
    """
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.misclassified_data = []
        self.misclassified_ground_truth = []
        self.misclassified_predicted = []

    def get_misclassified(self, number=20):
        # predict_generator
        '''
            Generates output predictions for the input samples.
            predict(x, batch_size=None, verbose=0, steps=None, callbacks=None,
            max_queue_size=10, workers=1, use_multiprocessing=False)
        '''
        if self.data_loader:
            data_loader = self.data_loader

        misclassified_data = []
        misclassified_ground_truth = []
        misclassified_predicted = []
        self.model.eval()

        with torch.no_grad():
            # TODO : None object
            for data, target in data_loader:
                # move to respective device
                data, target = data.to(self.device), target.to(self.device)
                # inference
                output = self.model(data)

                # get predicted output and the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)

                # get misclassified list for this batch
                misclassified_list = (target.eq(pred.view_as(target)) == False)
                misclassified = data[misclassified_list]
                ground_truth = pred[misclassified_list]
                predicted = target[misclassified_list]

                # stitching together
                misclassified_data.append(misclassified)
                misclassified_ground_truth.append(ground_truth)
                misclassified_predicted.append(predicted)

                # stop after enough false positives
                if len(misclassified_data) >= number:
                    break

        # converting to torch tensors
        misclassified_data = torch.cat(misclassified_data)
        misclassified_ground_truth = torch.cat(misclassified_ground_truth)
        misclassified_predicted = torch.cat(misclassified_predicted)

        self.misclassified_data = misclassified_data
        self.misclassified_ground_truth = misclassified_ground_truth
        self.misclassified_predicted = misclassified_predicted

        return misclassified_data, misclassified_ground_truth, misclassified_predicted

    # def plot_miscls(self, mean=CIFAR10Data.mean, std=CIFAR10Data.std):
    #     misclassified_images, ground_truth, predicted = self.get_misclassified(number)
    #
    #     misclassified_images = misclassified_images.cpu()
    #     # convert BCHW to BHWC for plotting
    #     misclassified_images = misclassified_images.permute(0, 2, 3, 1)


    def classwise_accuracy(self, classes=CIFAR10Data.classes):
        '''
            Class wise total accuracy
        :param classes:
        :return:
        '''
        class_total = list(0. for i in range(10))
        class_correct = list(0. for i in range(10))

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

    def plot_misclassified(self, labels = [], number=20, ncols=4, figsize=(6.0, 4.0), title='', mean=CIFAR10Data.mean, std=CIFAR10Data.std):
        if self.data_loader:
            data_loader = self.data_loader
        if not self.misclassified_data:
            misclassified_images, ground_truth, predicted = self.get_misclassified(number)
        else:
            misclassified_images = self.misclassified_data
            ground_truth = self.misclassified_ground_truth
            predicted = self.misclassified_predicted

        print('shapes : ', (misclassified_images.shape), len(ground_truth), type(predicted))

        misclassified_images1 = misclassified_images.cpu()
        misclassified_images = misclassified_images.cpu()
        # convert BCHW to BHWC for plotting
        misclassified_images1 = misclassified_images1.permute(0, 2, 3, 1)
        misclassified_images = misclassified_images.permute(0, 2, 3, 1)
        # Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]
        # As per discussions in pytorch forums, torch only supports NCHW format.
        # Use permute method to swap the axis.

        nrows = (number//ncols) + 1
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        axes = ax.flatten()
        # super title
        fig.suptitle(title)
        unorm = UnNormalize(mean=mean, std=std)

        for idx in range(len(misclassified_images1)):
            if idx <= number:
                img = misclassified_images[i, :, :, :]
                img = unorm(img.permute(2, 0, 1)).permute(1, 2, 0)
                img = (img.numpy() * 255).astype(np.uint8)
                if labels:
                    title = "GT:{}\nPred:{}".format(labels[int(ground_truth[idx])], labels[int(predicted[idx])])
                else:
                    title = "GT:{}\nPred:{}".format(ground_truth[idx].numpy(), predicted[idx].numpy())
                axes[idx].imshow(img, interpolation='bilinear')
                axes[idx].set_title(title)
            else:
                break
        # Hide empty subplot boundaries
        [ax.set_visible(False) for ax in axes[idx + 1:]]
        plt.show()
        print('-+-+=+=+*+*'*100)
        for i in range((nrows*ncols)):
            plt.subplot(nrows,ncols,i+1)
            plt.tight_layout()
            # transposing as the channels are first here and plt expects them to be last
            img = misclassified_images[i, :, :, :]
            img = unorm(img.permute(2, 0, 1)).permute(1, 2, 0)
            plt.imshow((img.numpy() * 255).astype(np.uint8), interpolation='bilinear')

            if labels:
                plt.title("GT:{}\nPred:{}".format(labels[int(ground_truth[i])], labels[int(predicted[i])]))
            else:
                plt.title("GT:{}\nPred:{}".format(ground_truth[i].numpy(), predicted[i].numpy()))
            plt.xticks([])
            plt.yticks([])


# def make_grid(images, labels, rows=0, cols=3):
#
#     rc = {"axes.spines.left" : False,
#       "axes.spines.right" : False,
#       "axes.spines.bottom" : False,
#       "axes.spines.top" : False,
#       "axes.grid" : False,
#       "xtick.bottom" : False,
#       "xtick.labelbottom" : False,
#       "ytick.labelleft" : False,
#       "ytick.left" : False}
#     plt.rcParams.update(rc)
#
#     # expected input - list of numpy array with shape as - height, width, channels
#     total = len(images)
#     if not rows:
#         rows = int(math.ceil(float(total) / cols))
#     if not images:
#         print('No images')
#         # exit()
#     channels = images[0].shape[2]
#     if not channels:
#         image = np.expanddims(image, axis=2)
#     print(rows)
#     fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(2*rows, 2*cols))
#     fig.subplots_adjust(hspace=0.01, wspace=0.01)
#     ax = axes.flatten()
#     for idx in range(total):
#         image = images[idx]
#         if len(image.shape) == 2 : # if no channel
#             image = image.expanddims(image, axis=2)
#         ax[idx].imshow(img, interpolation='bilinear')
#         ax[idx].title(labels[idx])
#     plt.show()
