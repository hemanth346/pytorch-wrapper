import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# plt.style.use("dark_background")


class Plotter(object):
    pass

    def plot_history(train_history, test_history):
        (train_acc, train_losses) = train_history
        (test_acc, test_losses) = test_history

        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Model history')
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")
        plt.show()

    def plot_misclassified(learner, labels=[], number=20, cols=4, title='', mean=CIFAR10Data.mean, std=CIFAR10Data.std):
        if self.data_loader:
            data_loader = self.data_loader

        misclassified_images, ground_truth, predicted = self.get_misclassified(number)

        print('shapes : ', type(misclassified_images), type(ground_truth), type(predicted))

        misclassified_images = misclassified_images.cpu()

        # convert BCHW to BHWC for plotting
        misclassified_images = misclassified_images.permute(0, 2, 3, 1)
        # Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]
        # As per discussions in pytorch forums, torch only supports NCHW format.
        # Use permute method to swap the axis.

        rows = (number // cols) + (number % cols)
        fig = plt.figure()
        fig.set_figheight(rows * 3)
        fig.set_figwidth(cols * 3)

        # super title
        fig.suptitle(title)
        unorm = UnNormalize(mean=mean, std=std)

        for i in range((rows * cols)):
            plt.subplot(rows, cols, i + 1)
            plt.tight_layout()
            # transposing as the channels are first here and plt expects them to be last
            img = misclassified_images[i, :, :, :]
            img = unorm(img.permute(2, 0, 1)).permute(1, 2, 0)
            plt.imshow((img.numpy() * 255).astype(np.uint8), interpolation='bilinear')

            if labels:
                plt.title("GT:{}, Pred:{}".format(labels[int(ground_truth[i])], labels[int(predicted[i])]))
            else:
                plt.title("GT:{}, Pred:{}".format(ground_truth[i].numpy(), predicted[i].numpy()))
            plt.xticks([])
            plt.yticks([])

    @staticmethod
    def make_grid(nrows, ncols=3, figsize=(6.0, 4.0)):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        axes = ax.flatten()
        return axes

    @staticmethod
    def set_plt_param():
        rc = {"axes.spines.left": False,
              "axes.spines.right": False,
              "axes.spines.bottom": False,
              "axes.spines.top": False,
              "axes.grid": False,
              "xtick.bottom": False,
              "xtick.labelbottom": False,
              "ytick.labelleft": False,
              "ytick.left": False}
        plt.rcParams.update(rc)
