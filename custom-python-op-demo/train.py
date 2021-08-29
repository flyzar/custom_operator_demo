import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.jit import ProgramTranslator

from model import LeNet

import argparse
import time

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

def train(args):
    # set device
    device = 'gpu:0' if paddle.is_compiled_with_cuda() and args.device == 'GPU' else 'cpu'
    paddle.set_device(device)

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = LeNet()
    model.train()

    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    time_start = time.time()
    for epoch in range(args.epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()

    cost = time.time() - time_start
    print('train time: %.3f s.' % cost)
    


def parse_args():
    parser = argparse.ArgumentParser("mnist model")
    # parser.add_argument(
    #     '--to_static', type=bool, default=True, help='whether to train model in static mode.')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--epochs', type=int, default=2, help='The epochs of train.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # train in dygraph mode
    train(args)
