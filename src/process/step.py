import torch
from ..utils.objects import stats

'''
def softmax_accuracy(probs, all_labels):
    acc = (torch.argmax(probs) == all_labels).sum()
    acc = torch.div(acc, len(all_labels) + 0.0)
    return acc
'''
def softmax_accuracy(probs, all_labels):
    predictions = (probs > 0.5).long()
    all_labels = all_labels.long()
    correct = (predictions == all_labels).sum().float()
    accuracy = correct / len(all_labels)
    return accuracy


class Step:
    # Performs a step on the loader and returns the result
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer

    def __call__(self, i, x, y):
        out = self.model(x)
        loss = self.criterion(out, y.float())
        acc = softmax_accuracy(out, y.float())

        if self.model.training:
            # calculates the gradient
            loss.backward()
            # and performs a parameter update based on it
            self.optimizer.step()
            # clears old gradients from the last step
            self.optimizer.zero_grad()

        # print(f"\tBatch: {i}; Loss: {round(loss.item(), 4)}", end="")
        return stats.Stat(out.tolist(), loss.item(), acc.item(), y.tolist())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
