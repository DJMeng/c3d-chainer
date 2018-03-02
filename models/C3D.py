from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


class Block3D(chainer.Chain):

    """A 3d convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, in_channels, out_channels, ksize, pad=1):
        super(Block3D, self).__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(ndim=3, in_channels=in_channels, out_channels=out_channels, ksize=ksize, pad=pad)

    def __call__(self, x, batch_norm):
        h = self.conv(x)
        h = batch_norm(h)
        return F.relu(h)


class C3D(chainer.Chain):

    """A C3D network.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=12):
        super(C3D, self).__init__()
        with self.init_scope():
            self.conv1a = Block3D(3, 64, 3)
            self.bn1a = L.BatchNormalization(64)
            # self.scale1a, self.shift1a = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv2a = Block3D(64, 128, 3)
            self.bn2a = L.BatchNormalization(128)
            # self.scale2a, self.shift2a = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv3a = Block3D(128, 256, 3)
            self.bn3a = L.BatchNormalization(256)
            # self.scale3a, self.shift3a = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv3b = Block3D(256, 256, 3)
            self.bn3b = L.BatchNormalization(256)
            # self.scale3b, self.shift3b = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv4a = Block3D(256, 512, 3)
            self.bn4a = L.BatchNormalization(512)
            # self.scale4a, self.shift4a = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv4b = Block3D(512, 512, 3)
            self.bn4b = L.BatchNormalization(512)
            # self.scale4b, self.shift4b = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv5a = Block3D(512, 512, 3)
            self.bn5a = L.BatchNormalization(512)
            # self.scale5a, self.shift5a = chainer.Variable(1.0), chainer.Variable(0.0)
            self.conv5b = Block3D(512, 512, 3)
            self.bn5b = L.BatchNormalization(512)
            # self.scale5b, self.shift5b = chainer.Variable(1.0), chainer.Variable(0.0)
            # self.fc5 = L.Linear(16384, 8192)
            self.fc6 = L.Linear(8192, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, class_labels)

    def __call__(self, x):
        h = self.conv1a(x, self.bn1a)
        h = F.max_pooling_nd(h, ksize=(1, 2, 2), stride=(1, 2, 2))
        h = self.conv2a(h, self.bn2a)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv3a(h, self.bn3a)
        h = self.conv3b(h, self.bn3b)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv4a(h, self.bn4a)
        h = self.conv4b(h, self.bn4b)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        h = self.conv5a(h, self.bn5a)
        h = self.conv5b(h, self.bn5b)
        h = F.max_pooling_nd(h, ksize=(2, 2, 2), stride=(2, 2, 2))
        # h = self.fc5(h)
        # h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc6(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc7(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc8(h)
        return h
        # return F.softmax(h)
        # return F.sigmoid(h)

