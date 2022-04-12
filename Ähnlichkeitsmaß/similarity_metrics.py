import re
import json
import os
import sys
import math
import random
import tarfile
import numpy as np
import scipy
from scipy import spatial
import tensorflow as tf
import tensorflow.keras as keras

# Funktionen der Intersection over Union und Hausdorffer Distanz, genauere Informationen enthält das Notebook bzw. können in der Masterarbeit nachgelesen werden.

def iou(a, b):
    m = keras.metrics.MeanIoU(num_classes=2)
    m.update_state(a, b)

    return m.result()

# Hausdorffer Distant nimmt zwei Array mit zweidimensionaler Art an, also müssen die Arrays vorverarbeitet werden, um damit berechnet zu werden.
def haussdorfer_dist(a, b):
    return scipy.spatial.distance.directed_hausdorff(a, b, seed=None)[0]