import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
from scipy import ndimage
import mlflow

class augment:
    def blur(self ,img):
        kernel_size = (7, 7)
        return cv2.GaussianBlur(img, kernel_size, 0)

    def contrast(self, img):
        img = Image.fromarray(img)
        contrast = np.random.randint(10, 30)
        contrast_img = img.point(lambda p: p * (contrast / 127 + 1) - contrast)
        return np.array(contrast_img)

    def bright(self, img):
        img = Image.fromarray(img)
        brightness = 1 + np.random.randint(1, 9) / 10
        return np.array(img.point(lambda p: p * brightness))

    def dark(self, img):
        return np.array(img * (np.random.randint(1, 9) / 10))

    def noised(self, img):
        rows, cols = img.shape[:2]
        noise_img = img.copy()
        for _ in range(1500):
            x, y = np.random.randint(0, rows), np.random.randint(0, cols)
            noise_img[x, y] = np.random.randint(80, 180)
        return np.array(noise_img)

    def sharp(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def rotate_image(self, img, angle):
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)

def data_augmentation(aug, image):
    original_image = image.copy()
    output = []
    aug_types = ["blur", "contrast", "bright", "dark", "noised", "sharp"]
    
    for aug_type in aug_types:
        copy_img = image.copy()
        aug_img = getattr(aug, aug_type)(copy_img)
        output.append(aug_img)

    final_out_x, final_out_y = [], []
    for index,im in enumerate(output):
        rotation_angles = [70]
        final_out_x.append(cv2.resize(im, (256,128), interpolation = cv2.INTER_AREA))
        final_out_y.append(cv2.resize(original_image, (256,128), interpolation = cv2.INTER_AREA))
        for rotation_angle in rotation_angles:
            copy_img = im.copy()
            aug_img = ndimage.rotate(copy_img, rotation_angle, mode = 'constant',cval=255.0)
            aug_img = cv2.resize(aug_img, (256,128), interpolation = cv2.INTER_AREA)
            final_out_x.append(aug_img)
            rotated_image = ndimage.rotate(original_image.copy(), rotation_angle, mode = 'constant',cval=255.0)
            final_out_y.append(cv2.resize(rotated_image, (256,128), interpolation = cv2.INTER_AREA))
    return final_out_x, final_out_y

# **MLflow Tracking**
mlflow.set_experiment("Data_Augmentation_Tracking")

with mlflow.start_run(run_name="Data_Preparation"):
    aug_types = ["blur", "contrast", "bright", "dark", "noised", "sharp", "rotate_image"]
    mlflow.log_param("Augmentation_Types", aug_types)

    root = 'Dataset/HR'
    width, height = 256, 128
    X_train, Y_train = [], []
    data_aug = augment()

    os.makedirs("augmented_images/X_images", exist_ok=True)
    os.makedirs("augmented_images/Y_images", exist_ok=True)

    total_files = len(os.listdir(root))  
    processed_files = 0

    for file_index, file in tqdm(enumerate(os.listdir(root)), total=total_files, desc="Processing Images"):
        file_path = os.path.join(root, file)
        original_image = cv2.imread(file_path)
        x, y = data_augmentation(data_aug, original_image)

        for train_x, train_y in zip(x, y):
            lr_2x = cv2.resize(train_x, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
            lr_1x = cv2.resize(lr_2x, (width, height), interpolation=cv2.INTER_AREA)
            X_train.append(lr_1x)
            Y_train.append(train_y)

        processed_files += 1
        if processed_files % 10 == 0:  # Log progress every 10 images
            mlflow.log_metric("Processed_Files", processed_files)

    X_train = np.array(X_train)
    np.save('X_train.npy', X_train)
    Y_train = np.array(Y_train)
    np.save('Y_train.npy', Y_train)

    mlflow.log_metric("Total_Images", X_train.shape[0])
    mlflow.log_artifact('X_train.npy')
    mlflow.log_artifact('Y_train.npy')

    print(f"X_train.shape: {X_train.shape} and Y_train.shape: {Y_train.shape}")
    print("Data Preparation Done ✅")

    print("Saving Augmented Data...")
    for index in tqdm(range(X_train.shape[0]), desc="Saving Images"):
        cv2.imwrite(f'augmented_images/X_images/{index}.png', X_train[index])
        cv2.imwrite(f'augmented_images/Y_images/{index}.png', Y_train[index])

        if index % 50 == 0:  # Log progress every 50 images
            mlflow.log_metric("Images_Saved", index)

    # Logging Augmented Images to MLflow
    mlflow.log_artifacts("augmented_images/X_images", artifact_path="X_images")
    mlflow.log_artifacts("augmented_images/Y_images", artifact_path="Y_images")

    print("MLflow Logging Complete ✅")
