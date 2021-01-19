import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from model import DANN, ADDA
import scipy.io as scio
from load_data import load_mat

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def test(model, nx, ny, source=True):
    alpha = 0
    model.eval()
    n_correct = 0
    with torch.no_grad():
        x_tar = torch.tensor(nx, dtype=torch.float32).to(DEVICE)
        y_tar = torch.tensor(ny, dtype=torch.float32).to(DEVICE)
        class_output, _ = model(input_data=x_tar, alpha=alpha, source=source)
        prob, pred = torch.max(class_output.data, 1)
        n_correct += (pred == y_tar.long()).sum().item()

    acc = float(n_correct) / len(nx) * 100
    return acc


def train(model, optimizer, x_src, y_src, x_tar, y_tar):
    loss_class = torch.nn.CrossEntropyLoss()
    nepoch = 100
    gamma = 0.5
    for epoch in range(nepoch):
        model.train()
        p = float(epoch) / nepoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        class_output, err_s_domain = model(input_data=x_src, alpha=alpha)
        err_s_label = loss_class(class_output, y_src.long())
        _, err_t_domain = model(input_data=x_tar, alpha=alpha, source=False)
        err_domain = err_t_domain + err_s_domain
        if model.get_name() == "DANN":
            err = err_s_label + gamma * err_domain
        else:
            err = err_domain
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: ' \
                  '{:.4f}, total_loss: {:.4f}'.format(
            epoch, nepoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item())
        print(item_pr)

        acc_src = test(model, x_src, y_src)
        acc_tar = test(model, x_tar, y_tar, source=False)
        test_info = 'Source acc: {:.4f}, target acc: {:.4f}'.format(acc_src, acc_tar)
        print(test_info)


if __name__ == '__main__':
    torch.random.manual_seed(10)

    train_data = scio.loadmat('./2020/train/1.mat')['de_feature']
    train_label = scio.loadmat('./2020/train/1.mat')['label']
    test_data = scio.loadmat('./2020/test/11.mat')['de_feature']
    test_label = scio.loadmat('./2020/test/11.mat')['label']

    train_data = (train_data - train_data.min(axis=0)) / (train_data.max(axis=0) - train_data.min(axis=0))
    test_data = (test_data - test_data.min(axis=0)) / (test_data.max(axis=0) - test_data.min(axis=0))
    train_label = np.array(train_label).reshape(len(train_label))
    test_label = np.array(test_label).reshape(len(test_label))

    train_data = torch.tensor(train_data, dtype=torch.float32).to(DEVICE)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(DEVICE)
    train_label = torch.tensor(train_label, dtype=torch.float32).to(DEVICE)
    test_label = torch.tensor(test_label, dtype=torch.float32).to(DEVICE)

    # train_data, train_label, test_data, test_label = load_mat()

    # model = ADDA(DEVICE, train_data, train_label).to(DEVICE)
    model = DANN(DEVICE).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train(model, optimizer, train_data, train_label, test_data, test_label)
