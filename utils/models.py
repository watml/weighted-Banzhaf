import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_feature, out_feature)

    def forward(self, X):
        y_pred = torch.sigmoid(self.linear(X))
        return y_pred