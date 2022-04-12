import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import datetime
from sklearn import model_selection
from natsort import natsorted
import argparse

# Training der Baseline

# Angabe der zugewiesenen GPU auf dem ML0-Server der TH-Nürnberg
os.environ["CUDA_VISIBLE_DEVICES"]="4"

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=2)
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument("--data_preprocessed", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, required=False)
parser.add_argument("--log_path", type=str, required=True)
parser.add_argument("--predicted_model_path", type=str, required=True)

args = parser.parse_args()

# Laden der Daten
def construct_file_path_list_from_dir(dir, file_filter):
    if isinstance(file_filter, str):
        file_filter = [file_filter]
    paths = [[] for _ in range(len(file_filter))]

    for root, _, files in os.walk(dir):
        for f_name in files:
            for i, f_substr in enumerate(file_filter):
                if f_substr in f_name:
                    (paths[i]).append(root + '/' + f_name)

    for i, p in enumerate(paths):
        paths[i] = natsorted(p)

    if len(file_filter) == 1:
        return paths[0]

    return tuple(paths)

def load_preprocessed_dataset():
    data_preprocessed_dir = args.data_preprocessed
    data_all = sorted(construct_file_path_list_from_dir(data_preprocessed_dir, ["_x.npy"]))
    label_all = sorted(construct_file_path_list_from_dir(data_preprocessed_dir, ["_y.npy"]))
    return np.array(data_all), np.array(label_all)

data, label = load_preprocessed_dataset()

def train_val_test_split(data, label, split=0.1):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=split)  
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=split)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(data, label)


def load_npy(npy_path):
    ret = []
    for p in npy_path:
        ret.append(np.load(p))
    return np.stack(ret)

class Batch_Load_Generator(keras.utils.Sequence) :
    def __init__(self, X, y, batch_size) :
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.X) * 24 / float(self.batch_size * 24))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.X[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
        
        output_X = []
        output_y = []
        
        img_batch_X = load_npy(batch_x)
        for index ,imgs in enumerate(img_batch_X):
            for img in imgs:
                img = img[:, :, 0:3]
                output_X.append(img)
                
                vox = np.load(batch_y[index])
                vox = np.argmax(vox, axis=-1)
                vox = vox.transpose(2, 0, 1)
                output_y.append(vox)
                
        return (np.array(output_X), np.array(output_y))


def voxel_loss_function(y, y_predicted):
    return tf.nn.softmax_cross_entropy_with_logits(y, y_predicted, axis=3)

def intersection_over_union(y, y_predicted):
    y_predicted = tf.where(y_predicted >= 1, 1, 0)
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y, y_predicted)
    return m.result()

encoder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(137, (2, 2), padding="same", input_shape=(137,137,3), kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (2, 2), padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(8, (3, 3), strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.2)),
    
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
])


decoder = tf.keras.Sequential([
    tf.keras.layers.Reshape((8, 8, 8, 1), input_shape=(8, 8, 8)),
    
    
    tf.keras.layers.Conv3DTranspose(16, (3, 3, 3), padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.keras.activations.relu),
    
    tf.keras.layers.Conv3DTranspose(32, (3, 3, 3), strides= 2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.keras.activations.relu),
    
    tf.keras.layers.Conv3DTranspose(64, (3, 3, 3), strides= 2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.keras.activations.relu),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv3DTranspose(1, (3, 3, 3), activation="sigmoid", padding ="same", kernel_initializer=tf.keras.initializers.GlorotNormal()),
    tf.keras.layers.Reshape([32,32,32])
    
])


model = keras.models.Sequential([encoder, decoder])

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=voxel_loss_function, metrics=[tf.keras.metrics.Accuracy(), intersection_over_union], run_eagerly=True)


training_batch_generator = Batch_Load_Generator(X_train, y_train, args.batchsize)
valid_batch_generator = Batch_Load_Generator(X_val, y_val, args.batchsize)


checkpoint_path = "training/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    period=100,
                                                    verbose=1)

# Wurde für Experimente verwendet; Callback: Early Stopping, dabei wird der Validation Loss im Blick gehalten und das Training frühzeitig 
# gestoppt, falls der Loss des Validation Sets sich verschlechtert oder gar keine Verbesserung mehr stattfindet
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

log_dir = args.log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(training_batch_generator,
                epochs=args.epochs, validation_data=valid_batch_generator, 
                callbacks=[cp_callback, tensorboard_callback])
