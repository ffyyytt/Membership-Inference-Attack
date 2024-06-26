import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import torch
import dataCIFARonline
from utils import *
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms

tf.get_logger().setLevel('INFO')

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
    
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    tf.config.set_soft_device_placement(True)

    print('Running on TPU ', tpu.master())
elif len(gpus) > 0:
    strategy = tf.distribute.MirroredStrategy(gpus)
    print('Running on ', len(gpus), ' GPU(s) ')
else:
    strategy = tf.distribute.get_strategy()
    print('Running on CPU')

print("Number of accelerators: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE

seedBasic()
backbone = "ResNet50"
device = "tf"
cenTrainDataLoader = dataCIFARonline.loadCenTrainCIFAR10(device)
miaDataLoader, miaLabels, memberLabels, inOutLabels = dataCIFARonline.loadMIADataCIFAR10(device)

cenModel = trainTFModel(cenTrainDataLoader, strategy, dataCIFARonline.__CIFAR10_N_CLASSES__, backbone)
yPred = [cenModel.predict(miaDataLoader), miaLabels]

print(yPred[0].shape, yPred[1].shape)
print(np.argmax(yPred[0], axis=1))
print(np.argmax(yPred[1], axis=1))
print(np.mean(np.argmax(yPred[0], axis=1) == np.argmax(yPred[1], axis=1)))

shadowPreds = []
shadowModels = []
for i in range(dataCIFARonline.__CIFAR10_N_SHADOW__):
    shadowDataLoader = dataCIFARonline.loadCenShadowTrainCIFAR10(i, device)
    shadowModels.append(trainTFModel(shadowDataLoader, strategy, dataCIFARonline.__CIFAR10_N_CLASSES__, backbone))
    shadowPreds.append([shadowModels[-1].predict(miaDataLoader), miaLabels])
    gc.collect()
    torch.cuda.empty_cache()
    if (i > 10):
        scores = computeMIAScore(yPred, shadowPreds, inOutLabels)
        print(f"Attack: {roc_auc_score(memberLabels, scores)}")
        print(f"TPR at {0.001} FPR: {TPRatFPR(memberLabels, scores, 0.001)}")

print(inOutLabels)
scores = computeMIAScore(yPred, shadowPreds, inOutLabels)
print(f"Attack: {roc_auc_score(memberLabels, scores)}")
for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    print(f"TPR at {thr} FPR: {TPRatFPR(memberLabels, scores, thr)}")