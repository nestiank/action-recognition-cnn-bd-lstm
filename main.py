#
# Video Action Recognition with Pytorch
#
# Paper citation
#
# Action Recognition in Video Sequences using
# Deep Bi-Directional LSTM With CNN Features
# 2017, AMIN ULLAH et al.
# Digital Object Identifier 10.1109/ACCESS.2017.2778011 @ IEEEAccess
#

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split

import time
import matplotlib.pyplot as plt
import math
import pickle

from dataset import prepare_dataset
from dataset import ToFloatTensorInZeroOne
from model import LSTM_with_CNN

# For updating learning rate
def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_and_eval(colab, batch_size, done_epochs, train_epochs):
    # Preprocessing dataset
    transform_dataset_train = transforms.Compose([
        ToFloatTensorInZeroOne(),
        transforms.Resize([128, 171]),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop(112)
    ])
    transform_dataset_test  = transforms.Compose([
        ToFloatTensorInZeroOne(),
        transforms.Resize([128, 171]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.CenterCrop(112)
    ])

    # Preparing dataset
    if colab:
        root = '/content/drive/MyDrive/dataset/video'
        annotation_path = '/content/drive/MyDrive/dataset/annotation'
    else:
        root = './dataset/video'
        annotation_path = './dataset/annotation'

    dataset_train = torchvision.datasets.HMDB51(root=root,
                                                annotation_path=annotation_path,
                                                frames_per_clip=10,
                                                step_between_clips=5,
                                                train=True,
                                                transform=transform_dataset_train)
    dataset_test  = torchvision.datasets.HMDB51(root=root,
                                                annotation_path=annotation_path,
                                                frames_per_clip=10,
                                                step_between_clips=5,
                                                train=False,
                                                transform=transform_dataset_test)

    # Train set 52.5%, validation set 17.5%, test set 30%
    dataset_len = len(dataset_train)
    train_len = math.floor(dataset_len * 0.75)
    val_len = dataset_len - train_len
    dataset_train, dataset_val = random_split(dataset_train, [train_len, val_len])

    # Loading dataset
    loader_train  = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True)
    loader_val    = torch.utils.data.DataLoader(dataset=dataset_val,
                                                batch_size=batch_size,
                                                shuffle=True)
    loader_test   = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preparing checkpoint location
    if colab:
        base_path = '/content/drive/MyDrive/checkpoints'
        pickle_path = '/content/drive/MyDrive/results'
    else:
        base_path = './checkpoints'
        pickle_path = './results'

    # Loading model
    model = LSTM_with_CNN().to(device)
    if done_epochs > 0:
        checkpoint = torch.load(base_path + '/lstm_epoch' + str(done_epochs) + '.ckpt', map_location=device)
        model.load_state_dict(checkpoint)
        with open(base_path + '/history.pickle', 'rb') as fr:
            history = pickle.load(fr)
    else:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Hyperparameters
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    total_steps = len(loader_train)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(done_epochs, done_epochs + train_epochs):
        print('Train: Epoch {} start @ {}'.format(epoch + 1, time.strftime('%c', time.localtime(time.time()))))

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Train
        for batch_index, (videos, audios, labels) in enumerate(loader_train):
            # videos.shape = torch.Size([batch_size, frames_per_clip, 3, 112, 112])
            # labels.shape = torch.Size([batch_size])

            videos = videos.permute(0, 2, 1, 3, 4)
            # videos.shape = torch.Size([batch_size, 3, frames_per_clip, 112, 112])

            videos = videos.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train loss
            train_loss += loss.item()

            # Train accuracy
            value, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total

            if (batch_index + 1) % 20 == 0:
                print('Train: Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}'
                    .format(epoch + 1, done_epochs + train_epochs, batch_index + 1, total_steps, train_loss / (batch_index + 1)))
                print('Step train accuracy: {} %'.format(train_acc))

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        print('Train: Epoch [{}/{}] Loss: {:.4f}'.format(epoch + 1, done_epochs + train_epochs, train_loss))
        print('Train accuracy: {} %'.format(train_acc))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0

            for batch_index, (videos, audios, labels) in enumerate(loader_val):
                # videos.shape = torch.Size([batch_size, frames_per_clip, 3, 112, 112])
                # labels.shape = torch.Size([batch_size])

                videos = videos.permute(0, 2, 1, 3, 4)
                # videos.shape = torch.Size([batch_size, 3, frames_per_clip, 112, 112])

                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                loss = criterion(outputs, labels)

                # Validation loss
                val_loss += loss.item()

                # Validation accuracy
                value, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print('Validation: Epoch [{}/{}] Loss: {:.4f}'.format(epoch + 1, done_epochs + train_epochs, val_loss))
            print('Validation accuracy: {} %'.format(val_acc))

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            learning_rate /= 3
            update_learning_rate(optimizer, learning_rate)

        # Save checkpoint
        torch.save(model.state_dict(), base_path + '/lstm_epoch' + str(epoch + 1) + '.ckpt')

    # Test
    print('Test: evaluation start @ {}'.format(time.strftime('%c', time.localtime(time.time()))))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for batch_index, (videos, audios, labels) in enumerate(loader_test):
            # videos.shape = torch.Size([batch_size, frames_per_clip, 3, 112, 112])
            # labels.shape = torch.Size([batch_size])

            videos = videos.permute(0, 2, 1, 3, 4)
            # videos.shape = torch.Size([batch_size, 3, frames_per_clip, 112, 112])

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            value, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test accuracy: {} %'.format(100 * correct / total))

    print('Finished training @ {}'.format(time.strftime('%c', time.localtime(time.time()))))

    plt.subplot(2, 1, 1)
    plt.plot(range(epoch + 1), history['train_loss'], label='Train loss', color='red')
    plt.plot(range(epoch + 1), history['val_loss'], label='Validation loss', color='blue')

    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(epoch + 1), history['train_acc'], label='Train accuracy', color='red')
    plt.plot(range(epoch + 1), history['val_acc'], label='Validation accuracy', color='blue')

    plt.title('Accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplots(constrained_layout=True)

    if colab:
        base_path = '/content/drive/MyDrive/results'
    else:
        base_path = './results'

    plt.savefig(base_path + '/result' + time.strftime('_%Y%m%d_%H%M%S', time.localtime(time.time())) + '.png')

    with open(pickle_path + '/history.pickle','wb') as fw:
        pickle.dump(history, fw)

if __name__ == '__main__':
    # Set True when using Google Colab
    colab = False

    # Consider GPU memory limit
    # Paper suggested 512
    # Suggest 128 for GTX 1660 Ti
    batch_size = 128

    # Last checkpoint's training position
    done_epochs = 0

    # Consider Google Colab time limit
    # How much epochs to train now
    train_epochs = 8

    prepare_dataset(colab)
    train_and_eval(colab, batch_size, done_epochs, train_epochs)
