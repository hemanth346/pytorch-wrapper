import os
import torch 
from tqdm.auto import tqdm, trange

__all__ = ['Trainer']

class Trainer(Object):
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = loss_fn
        self.optimzier = optimizer
        self.train_accuracy = []
        self.train_loss = []
        self.test_accuracy = []
        self.test_loss = []
    
    def _train_epoch(self, epoch):

        loss_history = []
        accuracy_history = []

        self.model.train()  # set the model in training mode

        train_loss = 0
        correct = 0
        total = 0
        processed = 0

        pbar = tqdm(self.train_loader, dynamic_ncols=True)

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output, target)

            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

            _, predicted = output.max(1)

            total += target.size(0)

            correct += predicted.eq(target).sum().item()

            processed += len(data)

            pbar.set_description(
                desc=f'epoch={epoch-1+batch_idx/len(pbar):.2f} | loss={train_loss/(batch_idx+1):.10f} | accuracy={100.*correct/total:.2f} {correct}/{total} | batch_id={batch_idx}')

            accuracy_history.append(100.*correct/processed)
            loss_history.append(loss.data.cpu().numpy().item())

        return (accuracy_history, loss_history)

    def _test_epoch(self, epoch):

        loss_history = []
        accuracy_history = []

        self.model.eval()  # set the model in evaluation mode

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # pbar.set_description(
                #     desc=f'epoch={epoch+batch_idx/len(pbar):.2f} | loss={test_loss/(batch_idx+1):.10f} | accuracy={100.*correct/total:.2f} {correct}/{total} | batch_id={batch_idx}')

        print(
            f'Test Set: Average Loss: {test_loss/len(self.test_loader):.8f}, Accuracy: {100. * correct / total:.2f} ({correct}/{total})')

        loss_history.append(test_loss/len(self.test_loader))
        accuracy_history.append((100. * correct) / total)

        return (accuracy_history, loss_history)

    def run(self, epochs):
        for epoch in range(1, epochs+1):
            print(f'\nEpoch: {epoch}')
            train_epoch_history = self._train_epoch(epoch)  # train this epoch
            self.train_accuracy.extend(train_epoch_history[0])
            self.train_loss.extend(train_epoch_history[1])

            test_epoch_history = self._test_epoch(epoch)  # test this epoch
            self.test_accracy.extend(test_epoch_history[0])
            self.test_loss.extend(test_epoch_history[1])

    def get_train_history(self):
        return (self.train_accuracy, self.train_loss)

    def get_test_history(self):
        return (self.test_accuracy, self.test_loss)


# class train():
#     loss_fn, history, predict, valuate, show_results, 

#     pass
