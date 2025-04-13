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
                                     UpSampling2D, BatchNormalization, Flatten, LeakyReLU, Dense, add, PReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from skimage import img_as_ubyte
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model
from numpy.random import randint

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Define GAN Generator and Discriminator

def res_block(ip):
    res_model = Conv2D(64, (3,3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    
    return add([ip, res_model])

def create_gen(gen_ip, num_res_block=16):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    temp = layers
    
    for _ in range(num_res_block):
        layers = res_block(layers)
    
    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, temp])
    
    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)

def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3,3), strides=strides, padding="same")(ip)
    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)
    disc_model = LeakyReLU(alpha=0.2)(disc_model)
    return disc_model

def create_disc(disc_ip):
    df = 64
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(disc_ip, validity)

# Load Dataset Paths
X_train_path = "X_train.npy"
Y_train_path = "Y_train.npy"
batch_size = 2

def data_generator(X_path, Y_path):
    X_data = np.load(X_path, mmap_mode='r')[:5000]
    Y_data = np.load(Y_path, mmap_mode='r')[:5000]
    def gen():
        for i in range(len(X_data)):
            x_sample = np.clip(X_data[i].astype(np.float32) / 255.0, 0.0, 1.0)
            y_sample = np.clip(Y_data[i].astype(np.float32) / 255.0, 0.0, 1.0)
            yield x_sample, y_sample  
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(128, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 256, 3), dtype=tf.float32)
        )
    )

train_ds = data_generator(X_train_path, Y_train_path).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = data_generator(X_train_path, Y_train_path).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_ds_size = sum(1 for _ in train_ds)
print(f"Train Dataset Size: {train_ds_size}")
# Initialize GAN
gen_input = Input(shape=(128, 256, 3))
disc_input = Input(shape=(128, 256, 3))
generator = create_gen(gen_input)
discriminator = create_disc(disc_input)

generator.summary()
discriminator.summary()
from tensorflow.keras.losses import Huber
huber = Huber(delta=1.0)
# Compile GAN Components
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=huber)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

#Edits by Pranav
class MLflowLogger(Callback):
    def on_epoch_end(self, epoch, logs = None):
        if logs is not None:
            for key,value in logs.items():
                mlflow.log_metric(key, value, step = epoch)

# Train Model
epochs = 25


model_type = "gan"
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"training_runs/{run_id}_{model_type}"
os.makedirs(output_dir, exist_ok=True)

mlflow.set_experiment("Image Enhancement Training")
with mlflow.start_run(run_name=f"Image Enhancement Training with {run_id}_{model_type}"):
    mlflow.log_params({"epochs": epochs, "batch_size": batch_size})
    
    callback = EarlyStopping(monitor='val_loss', patience=10)
    mlflow_logger = MLflowLogger()
    
    for epoch in range(epochs):
        count = 0
        for x_batch, y_batch in tqdm(train_ds,total=train_ds_size):
            generated_images = generator.predict(x_batch, verbose=0)
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            d_loss_real = discriminator.train_on_batch(y_batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = generator.train_on_batch(x_batch, y_batch)
            mlflow.log_metrics({"d_loss": d_loss[0], "g_loss": g_loss}, step=count)
            # print(f"Count : {count} - D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")
            count += 1
        print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")
        generator.save(os.path.join(output_dir, f"generator_epoch_{epoch + 1}.h5"))
        discriminator.save(os.path.join(output_dir, f"discriminator_epoch_{epoch + 1}.h5"))     
        mlflow.log_metrics({"d_loss": d_loss[0], "g_loss": g_loss}, step=epoch)
        # Save Generator and Discriminator
    generator.save(os.path.join(output_dir, 'generator.h5'))
    discriminator.save(os.path.join(output_dir, 'discriminator.h5'))
    mlflow.keras.log_model(generator, "generator")
    mlflow.keras.log_model(discriminator, "discriminator")

    print("Training Complete.")

    # Function to save training plots
    def save_plot(metric_values, val_metric_values, metric, title):
        plt.plot(metric_values, label=f'Training {metric}')
        plt.plot(val_metric_values, label=f'Validation {metric}')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        mlflow.log_artifact(os.path.join(output_dir, f'{metric}.png'))
        plt.close()

    # Placeholder lists for storing loss values (since history is not used)
    train_losses = []
    val_losses = []

    # Save loss plot manually
    save_plot(train_losses, val_losses, 'loss', 'Training and Validation Loss')

    metrics = {
        "train_per_pixel_loss": [], "val_per_pixel_loss": [],
        "train_perceptual_loss": [], "val_perceptual_loss": [],
        "train_psnr": [], "val_psnr": [],
        "train_ssim": [], "val_ssim": []
    }

    # Metric calculation functions
    def calculate_metrics(img1, img2):
        img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        mae = np.mean(np.abs(img1 - img2))
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
        ssim = np.mean((2 * img1 * img2 + 0.01 * 255) / (img1**2 + img2**2 + 0.01 * 255))
        return mae, mse, psnr, ssim

    def evaluate_dataset(dataset_x, dataset_y, metric_prefix):
        for i in range(len(dataset_x)):
            output = generator.predict(dataset_x[i].reshape(1, *dataset_x[i].shape), verbose=0)[0]
            output = img_as_ubyte(np.clip(output, 0, 1))
            img_y = img_as_ubyte(dataset_y[i])
            mae, mse, psnr, ssim = calculate_metrics(img_y, output)
            metrics[f"{metric_prefix}_per_pixel_loss"].append(mae)
            metrics[f"{metric_prefix}_perceptual_loss"].append(mse)
            metrics[f"{metric_prefix}_psnr"].append(psnr)
            metrics[f"{metric_prefix}_ssim"].append(ssim)

    # Convert datasets to numpy arrays
    train_x, train_y = zip(*[(x.numpy(), y.numpy()) for x, y in train_ds])
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    val_x, val_y = zip(*[(x.numpy(), y.numpy()) for x, y in val_ds])
    val_x = np.concatenate(val_x, axis=0)
    val_y = np.concatenate(val_y, axis=0)

    # Evaluate the datasets
    evaluate_dataset(train_x, train_y, "train")
    evaluate_dataset(val_x, val_y, "val")

    # Averaging the metrics before storing them
    metrics_avg = {k: np.mean(v) for k, v in metrics.items()}
    df_metrics = pd.DataFrame([metrics_avg])

    # Save and log metrics
    df_metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    mlflow.log_artifact(os.path.join(output_dir, "metrics.csv"))
    mlflow.log_metrics(metrics_avg)

    print("Metrics logged in MLflow and CSV.")
