import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     UpSampling2D, BatchNormalization,add)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from skimage import img_as_ubyte
from datetime import datetime
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD,Adam
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Load Dataset Paths
X_train_path = "X_train.npy"
Y_train_path = "Y_train.npy"

# Define batch size
batch_size = 8

# Data Generator Function
def data_generator(X_path, Y_path):
    X_data = np.load(X_path, mmap_mode='r')[:5000]  # Memory-map the dataset
    Y_data = np.load(Y_path, mmap_mode='r')[:5000]

    def gen():
        for i in range(len(X_data)):
            x_sample = np.clip(X_data[i].astype(np.float32) / 255.0, 0.0, 1.0)  # Normalize & Clip
            y_sample = np.clip(Y_data[i].astype(np.float32) / 255.0, 0.0, 1.0)
            yield x_sample, y_sample  

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
        )
    )



batch_size = 8
train_ds = data_generator("X_train.npy", "Y_train.npy").batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = data_generator("X_train.npy", "Y_train.npy").batch(batch_size).prefetch(tf.data.AUTOTUNE)
# Convert the Dataset objects to numpy arrays
train_x = []
train_y = []
val_x = []
val_y = []

# Iterate over the training dataset and collect the data
for x_batch, y_batch in train_ds:
    train_x.append(x_batch.numpy())
    train_y.append(y_batch.numpy())

# Iterate over the validation dataset and collect the data
for x_batch, y_batch in val_ds:
    val_x.append(x_batch.numpy())
    val_y.append(y_batch.numpy())

# Convert the lists to numpy arrays
train_x = np.concatenate(train_x, axis=0)
train_y = np.concatenate(train_y, axis=0)
val_x = np.concatenate(val_x, axis=0)
val_y = np.concatenate(val_y, axis=0)

# Optional: Normalize if not already done in the generator
train_x = train_x 
train_y = train_y
val_x = val_x
val_y = val_y 

# X_data = np.load("X_train.npy", mmap_mode="r")[:10000] / 255.0
# Y_data = np.load("Y_train.npy", mmap_mode="r")[:10000] / 255.0

# # Check for NaN or Inf values
# assert not np.isnan(X_data).any(), "NaN found in X_data"
# assert not np.isnan(Y_data).any(), "NaN found in Y_data"
# assert not np.isinf(X_data).any(), "Inf found in X_data"
# assert not np.isinf(Y_data).any(), "Inf found in Y_data"

# train_x, val_x, train_y, val_y = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
# Create Dataset Loaders
# train_x, val_x, train_y, val_y = train_test_split(np.arange(len(np.load(X_train_path, mmap_mode='r'))),
#                                                   np.arange(len(np.load(Y_train_path, mmap_mode='r'))),
#                                                   test_size=0.2)

# train_ds = data_generator(X_train_path, Y_train_path, batch_size)
# val_ds = data_generator(X_train_path, Y_train_path, batch_size)

# Custom MLflow Callback
class MLflowLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)  # Log metrics at each epoch

def step_decay(epoch):
    initial_rate = 1e-4  # Change from 1e-4 to 1e-5
    factor = int(epoch / 10)
    return initial_rate / (10 ** factor)
class NaNMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if np.isnan(logs["loss"]):
            print(f"Stopping training due to NaN loss at epoch {epoch}")
            self.model.stop_training = True

nan_monitor = NaNMonitor()

def build_model():
    input_img = Input(shape=(128, 256, 3))
    l01 = Conv2D(32, (5, 5), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(input_img)
    
    l02 = Conv2D(32, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l01)
    l03 = MaxPooling2D(2,padding='same')(l02)
    
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l03)
    
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l1)

    l3 = MaxPooling2D(padding='same')(l2)
    
    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l4)

    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l6)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu', 
                activity_regularizer=regularizers.l1(10e-10))(l7)
    # change dropout values
    l7 = Dropout(0.5)(l7)
    l8 = UpSampling2D()(l7)

    l9 = Conv2D(128, (3, 3), padding='same', activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l9)

    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l13)

    l15 = add([l14, l2])
    l16 = UpSampling2D()(l15)
    l17 = Conv2D(32, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l16)
    l18 = Conv2D(32, (3, 3), padding='same', activation='relu',
                 activity_regularizer=regularizers.l1(10e-10))(l17)
    l19 = add([l18, l02])
    decoded = Conv2D(3, (5, 5), padding='same', activation='relu', 
                     activity_regularizer=regularizers.l1(10e-10))(l19)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)
    model = Model(input_img, decoded)
    from tensorflow.keras.losses import Huber
    huber = Huber(delta=1.0)
    # model.compile(optimizer='adam', loss='mae',metrics=['accuracy'])
    model.compile(SGD(lr= 1e-1,momentum=0.9), loss=huber,metrics=['accuracy'])


    # input_layer = Input(shape=(None, None, 3))
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Dropout(0.5)(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = UpSampling2D((2, 2))(x)
    # output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same', dtype="float32")(x)
    # model = Model(inputs=[input_layer], outputs=[output_layer])
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)  # Clip gradients
    # from tensorflow.keras.losses import Huber
    # huber = Huber(delta=1.0)
    # model.compile(optimizer=optimizer, loss=huber, metrics=["accuracy"])

    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["accuracy"])

    return model

epochs = 50
batch_size = 8
model = build_model()
model.summary()
model_type = "unet_autoencoder"
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"training_runs/{run_id}_{model_type}"
os.makedirs(output_dir, exist_ok=True)

mlflow.set_experiment("Image Enhancement Training")
with mlflow.start_run(run_name=f"Image Enhancement Training with {run_id}_{model_type}"):
    mlflow.log_params({"epochs": epochs, "batch_size": batch_size})
    
    lr_schedule = LearningRateScheduler(step_decay)
    callback = EarlyStopping(monitor='val_loss', patience=10)
    mlflow_logger = MLflowLogger()
    # history = model.fit(train_x, train_y, validation_data=(val_x, val_y),
    #                     epochs=epochs, verbose=1,
    #                     callbacks=[lr_schedule, callback,mlflow_logger,nan_monitor],shuffle=True)
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, verbose=1,
                        callbacks=[lr_schedule, callback,mlflow_logger],shuffle=True)
    
    model.save(os.path.join(output_dir, 'model.h5'))
    mlflow.keras.log_model(model, "model")
    
    def save_plot(metric, title):
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        mlflow.log_artifact(os.path.join(output_dir, f'{metric}.png'))
        plt.close()
    
    save_plot('loss', 'Training and Validation Loss')
    save_plot('accuracy', 'Training and Validation Accuracy')
    
    metrics = {"train_per_pixel_loss": [], "val_per_pixel_loss": [],
               "train_perceptual_loss": [], "val_perceptual_loss": [],
               "train_psnr": [], "val_psnr": [],
               "train_ssim": [], "val_ssim": []}
    
    def calculate_metrics(img1, img2):
        img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        mae = np.mean(np.abs(img1 - img2))
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
        ssim = np.mean((2 * img1 * img2 + 0.01 * 255) / (img1**2 + img2**2 + 0.01 * 255))
        return mae, mse, psnr, ssim
    
    def evaluate_dataset(dataset_x, dataset_y, metric_prefix):
        for i in range(len(dataset_x)):
            output = model.predict(dataset_x[i].reshape(1, *dataset_x[i].shape), verbose=0)[0]
            output = img_as_ubyte(np.clip(output, 0, 1))
            img_y = img_as_ubyte(dataset_y[i])
            mae, mse, psnr, ssim = calculate_metrics(img_y, output)
            metrics[f"{metric_prefix}_per_pixel_loss"].append(mae)
            metrics[f"{metric_prefix}_perceptual_loss"].append(mse)
            metrics[f"{metric_prefix}_psnr"].append(psnr)
            metrics[f"{metric_prefix}_ssim"].append(ssim)
    
    evaluate_dataset(train_x, train_y, "train")
    evaluate_dataset(val_x, val_y, "val")
    
    # Averaging the metrics over the dataset before storing them
    metrics_avg = {k: np.mean(v) for k, v in metrics.items()}  # Average the metrics
    df_metrics = pd.DataFrame([metrics_avg])  # Create a DataFrame with a single row (average metrics)

    df_metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    mlflow.log_artifact(os.path.join(output_dir, "metrics.csv"))
    mlflow.log_metrics(metrics_avg)
    
    print("Training Complete. Metrics logged in MLflow and CSV.")
    print("Model saved in training_runs directory.")