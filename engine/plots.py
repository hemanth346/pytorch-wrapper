import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# plt.style.use("dark_background")

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
