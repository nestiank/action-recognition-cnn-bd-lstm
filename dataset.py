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
# See also main.py
#

import requests
import os
import glob
import torch

def download_file(URL, destination):
    session = requests.Session()
    response = session.get(URL, stream = True)
    save_response_content(response, destination)

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            # filter out keep-alive new chunks
            if chunk:
                f.write(chunk)

def prepare_dataset(colab):
    if colab:
        base_path = '/content/drive/MyDrive/dataset'
        checkpoints_path = '/content/drive/MyDrive/checkpoints'
        results_path = '/content/drive/MyDrive/results'
    else:
        base_path = './dataset'
        checkpoint_path = './checkpoints'
        results_path = './results'

    if not os.path.isdir(base_path):
        os.makedirs(base_path)
        os.makedirs(checkpoints_path)
        os.makedirs(results_path)

    if not os.path.isfile(base_path + '/hmdb51_org.rar'):
        print('Downloading the dataset...')
        download_file('http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar', base_path + '/hmdb51_org.rar')

    if not os.path.isdir(base_path + '/video'):
        print('Unraring the dataset...')
        os.makedirs(base_path + '/video')
        os.system('unrar e ' + base_path + '/hmdb51_org.rar ' + base_path + '/video')
        filenames = glob.glob(base_path + '/video/*.rar')
        for file_name in filenames:
            os.system(('unrar x %s ' + base_path + '/video') % file_name)

    if not os.path.isfile(base_path + '/test_train_splits.rar'):
        print('Downloading annotations of the dataset...')
        download_file('http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar', base_path + '/test_train_splits.rar')

    if not os.path.isdir(base_path + '/annotation'):
        print('Unraring annotations of the dataset...')
        os.makedirs(base_path + '/annotation')
        os.system('unrar e ' + base_path + '/test_train_splits.rar ' + base_path + '/annotation')
        filenames = glob.glob(base_path + '/annotation/*.rar')
        for file_name in filenames:
            os.system('unrar x %s ' + base_path + '/annotation' % file_name)  

def to_normalized_float_tensor(video):
    return video.permute(0, 3, 1, 2).to(torch.float) / 255

class ToFloatTensorInZeroOne(object):
    def __call__(self, video):
        return to_normalized_float_tensor(video)
