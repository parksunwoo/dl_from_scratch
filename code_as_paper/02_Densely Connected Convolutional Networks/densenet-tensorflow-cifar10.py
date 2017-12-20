"""Densely Connected Convolutional Networks, In CVPR 2017 (Best Paper Award)
video: https://www.youtube.com/watch?v=fe2Vn0mwALI&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS&index=29
reference site : https://github.com/liuzhuang13/DenseNet , https://github.com/YixuanLi/densenet-tensorflow """


import numpy as np
import tensorflow as tf
import argparse
import os


from tensorpack import *
from tensorpack.callbacks.param import ScheduledHyperParamSetter
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.symbolic_functions import  *
from tensorpack.tfutils.summary import *

from tensorpack.callbacks.inference import ScalarStats, ClassificationError
from tensorpack.callbacks.inference_runner import InferenceRunner
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.image import AugmentImageComponent
from tensorpack.dataflow.prefetch import PrefetchData
from tensorpack.graph_builder.model_desc import InputDesc, ModelDesc
from tensorpack.train.trainers import SyncMultiGPUTrainer
from tensorpack.trainv1.config import TrainConfig

BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self, depth):
        super(Model, self).__iniit__()
        self.N = int((depth -4) / 3)
        self.growthRate = 12

        def _get_inputs(self):
            return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                    InputDesc(tf.int32, [None], 'label')
                   ]


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x- pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x- pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/cifar-single-first%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate', [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])
        ],
        model=Model(depth=args.depth),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1', default=150, help='Epoch to drop learning rate to 0.01')
    parser.add_argument('--drop_2', default=225, help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--depth', default=40, help='the depth of denseNet')
    parser.add_argument('--max_epoch', default=300, help='max epoch')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()