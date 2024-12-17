from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    GlobalAveragePooling2D,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from werkzeug.utils import secure_filename
from mapping import sign_names, save_mapping
import random
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

# Ensure the upload and example folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("static/examples/", exist_ok=True)

# Global variables to store the model and data
model = None
data_loaded = False
X_train, X_val, y_train, y_val = None, None, None, None


# Function to load and preprocess data
def load_data():
    global X_train, X_val, y_train, y_val, data_loaded
    data_dir = "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
    num_classes = 43  # Number of classes in GTSRB

    images = []
    labels = []
    img_size = (64, 64)  # Increased image size

    logging.info("Loading and preprocessing training data...")

    for class_id in range(num_classes):
        class_dir = os.path.join(data_dir, f"{class_id:05d}")
        if not os.path.exists(class_dir):
            logging.warning(f"Directory {class_dir} does not exist. Skipping.")
            continue
        annotation_file = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
        if not os.path.exists(annotation_file):
            logging.warning(
                f"Annotation file {annotation_file} does not exist. Skipping."
            )
            continue
        with open(annotation_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_name = row["Filename"]
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Image {img_path} could not be read. Skipping.")
                    continue
                # Crop the image using the ROI
                try:
                    x1 = int(row["Roi.X1"])
                    y1 = int(row["Roi.Y1"])
                    x2 = int(row["Roi.X2"])
                    y2 = int(row["Roi.Y2"])
                    img_cropped = img[y1:y2, x1:x2]
                    img_resized = cv2.resize(img_cropped, img_size)
                    images.append(img_resized)
                    labels.append(int(row["ClassId"]))
                except Exception as e:
                    logging.error(f"Error processing image {img_path}: {e}")
                    continue

    images = np.array(images)
    labels = np.array(labels)
    logging.info(f"Total images loaded: {len(images)}")

    # Preprocess images using MobileNetV2 preprocessing
    images = preprocess_input(images)
    labels = to_categorical(labels, num_classes)

    # Check the shape of the images
    logging.info(f"Shape of images array: {images.shape}")  # Should be (N, 64, 64, 3)
    if images.shape[1:] != (64, 64, 3):
        logging.error(
            f"Unexpected image shape: {images.shape[1:]}. Expected (64, 64, 3)."
        )

    # Shuffle data
    from sklearn.utils import shuffle

    images, labels = shuffle(images, labels, random_state=42)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    logging.info(f"Training set shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    data_loaded = True
    logging.info("Data loading and preprocessing completed.")


# Function to retrain the model with new data, giving higher weight to corrected sample
def retrain_model(new_image, new_label):
    global model, X_train, y_train
    num_classes = 43

    logging.info("Starting model retraining with user feedback...")

    # Ensure data is loaded
    if not data_loaded:
        logging.info("Data not loaded. Loading data now...")
        load_data()

    # Create a small dataset for retraining
    # Over-sample the corrected sample
    num_copies = 100  # Number of times to duplicate the corrected sample
    new_images = np.array([new_image] * num_copies)
    new_labels = np.array(
        [to_categorical(new_label, num_classes=num_classes)] * num_copies
    )

    # Apply data augmentation to the new images
    datagen_new = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        horizontal_flip=False,
        fill_mode="nearest",
    )
    datagen_new.fit(new_images)

    # Use a subset of the original training data to prevent overfitting
    subset_size = 500
    if len(X_train) > subset_size:
        idx = np.random.choice(len(X_train), size=subset_size, replace=False)
        X_subset = X_train[idx]
        y_subset = y_train[idx]
    else:
        X_subset = X_train
        y_subset = y_train

    # Combine the new data with the subset
    X_retrain = np.concatenate((X_subset, new_images), axis=0)
    y_retrain = np.concatenate((y_subset, new_labels), axis=0)

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
    )
    datagen.fit(X_retrain)

    # Recompile the model with a higher learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Retrain the model
    history = model.fit(
        datagen.flow(X_retrain, y_retrain, batch_size=32),
        epochs=5,  # Increased number of epochs
        verbose=1,
    )

    # Save the updated model
    model.save("traffic_sign_model.h5")
    logging.info("Model retraining completed and saved.")


# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Route for training the model
@app.route("/train", methods=["GET", "POST"])
def train():
    global model, data_loaded
    if request.method == "POST":
        if not data_loaded:
            load_data()
        num_classes = 43

        logging.info("Building and compiling the model with MobileNetV2 base...")

        # Load the MobileNetV2 model pre-trained on ImageNet, exclude top layers
        base_model = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(64, 64, 3)
        )

        # Freeze the base model layers
        base_model.trainable = False

        # Add custom top layers
        inputs = Input(shape=(64, 64, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        logging.info("Starting model training...")

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            horizontal_flip=False,
            fill_mode="nearest",
        )
        datagen.fit(X_train)

        # Early stopping and learning rate reduction
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        )

        # Train the model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=64),
            epochs=30,  # Increased number of epochs
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        # Save the model
        model.save("traffic_sign_model.h5")

        logging.info("Model training completed and saved.")

        return render_template(
            "train.html", message="Model trained and saved successfully!"
        )
    else:
        return render_template("train.html")


# Route for testing the model
@app.route("/test", methods=["GET", "POST"])
def test():
    global model
    if model is None:
        if os.path.exists("traffic_sign_model.h5"):
            model = load_model("traffic_sign_model.h5")
            # Recompile the model
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        else:
            return render_template("test.html", message="Model is not trained yet.")

    if not data_loaded:
        load_data()

    logging.info("Evaluating the model on validation data...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    logging.info(f"Validation Accuracy: {val_acc}")

    return render_template(
        "test.html", accuracy=val_acc, report=report, sign_names=sign_names
    )


# Route for uploading and classifying an image
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global model
    if model is None:
        if os.path.exists("traffic_sign_model.h5"):
            model = load_model("traffic_sign_model.h5")
            # Recompile the model
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        else:
            return render_template("upload.html", message="Model is not trained yet.")

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", message="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", message="No selected file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess the image
            img = cv2.imread(filepath)
            if img is None:
                return render_template(
                    "upload.html", message="Cannot read uploaded image."
                )
            img_resized = cv2.resize(img, (64, 64))
            img_preprocessed = preprocess_input(img_resized)
            img_expanded = np.expand_dims(img_preprocessed, axis=0)

            # Verify the shape
            logging.info(
                f"Shape of uploaded image: {img_expanded.shape}"
            )  # Should be (1, 64, 64, 3)
            if img_expanded.shape[1:] != (64, 64, 3):
                logging.error(
                    f"Unexpected uploaded image shape: {img_expanded.shape[1:]}. Expected (64, 64, 3)."
                )
                return render_template(
                    "upload.html", message="Uploaded image has an incorrect shape."
                )

            # Predict the class
            prediction = model.predict(img_expanded)
            class_id = np.argmax(prediction)
            class_name = sign_names.get(str(class_id), "Unknown")

            # Get an example image from the predicted class
            class_dir = os.path.join("GTSRB/Final_Training/Images", f"{class_id:05d}")
            example_image_path = None
            if os.path.exists(class_dir):
                class_images = [
                    f
                    for f in os.listdir(class_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm"))
                ]
                if class_images:
                    example_image_name = random.choice(class_images)
                    example_image_full_path = os.path.join(
                        class_dir, example_image_name
                    )
                    # Copy the image to the static folder to display
                    example_image_dest = os.path.join(
                        "static", "examples", f"example_{class_id}.png"
                    )
                    if not os.path.exists(example_image_dest):
                        img_example = cv2.imread(example_image_full_path)
                        # Crop the image using the ROI
                        # Read the annotation for the example image
                        annotation_file = os.path.join(
                            class_dir, f"GT-{class_id:05d}.csv"
                        )
                        if os.path.exists(annotation_file):
                            with open(annotation_file, "r") as f:
                                reader = csv.DictReader(f, delimiter=";")
                                for row in reader:
                                    if row["Filename"] == example_image_name:
                                        try:
                                            x1 = int(row["Roi.X1"])
                                            y1 = int(row["Roi.Y1"])
                                            x2 = int(row["Roi.X2"])
                                            y2 = int(row["Roi.Y2"])
                                            img_example = img_example[y1:y2, x1:x2]
                                            break
                                        except Exception as e:
                                            logging.error(
                                                f"Error cropping example image {example_image_full_path}: {e}"
                                            )
                                            img_example = img_example  # Use the original image if cropping fails
                        img_example = cv2.resize(img_example, (100, 100))
                        cv2.imwrite(example_image_dest, img_example)
                    example_image_path = example_image_dest
            else:
                example_image_path = None

            return render_template(
                "upload.html",
                class_id=class_id,
                class_name=class_name,
                image_path=filepath,
                example_image_path=example_image_path,
                sign_names=sign_names,
            )
    elif request.method == "GET" and "feedback" in request.args:
        # Handle user feedback
        feedback = request.args.get("feedback")
        class_id = request.args.get("class_id", type=int)
        image_path = request.args.get("image_path")
        correct_class_id = request.args.get("correct_class_id", type=int)

        if feedback == "correct":
            logging.info("User confirmed the prediction was correct.")
            pass  # No action needed
        elif feedback == "incorrect" and correct_class_id is not None:
            logging.info(
                "User indicated the prediction was incorrect. Retraining the model."
            )
            correct_class_id = int(correct_class_id)
            # Ensure data is loaded before retraining
            if not data_loaded:
                load_data()
            # Load the image and preprocess it
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Cannot read uploaded image at {image_path}.")
                return redirect(
                    url_for("upload", message="Cannot read uploaded image.")
                )
            img_resized = cv2.resize(img, (64, 64))
            img_normalized = preprocess_input(img_resized)
            # Retrain the model with the new data
            retrain_model(img_normalized, correct_class_id)

        return redirect(url_for("upload", message="Thank you for your feedback!"))

    else:
        return render_template("upload.html", sign_names=sign_names)


# Route for adversarial attack
@app.route("/adversarial", methods=["GET", "POST"])
def adversarial_attack():
    global model
    if model is None:
        if os.path.exists("traffic_sign_model.h5"):
            model = load_model("traffic_sign_model.h5")
            # Recompile the model
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        else:
            return render_template(
                "adversarial.html", message="Model is not trained yet."
            )

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("adversarial.html", message="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("adversarial.html", message="No selected file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess the image
            img = cv2.imread(filepath)
            if img is None:
                return render_template(
                    "adversarial.html", message="Cannot read uploaded image."
                )
            img_resized = cv2.resize(img, (64, 64))
            img_preprocessed = preprocess_input(img_resized)
            img_expanded = np.expand_dims(img_preprocessed, axis=0)

            # Predict the original class
            prediction = model.predict(img_expanded)
            original_class_id = np.argmax(prediction)
            original_class_name = sign_names.get(str(original_class_id), "Unknown")

            # Convert model to Foolbox model
            fmodel = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs)
            fmodel = TensorFlowModel(fmodel, bounds=(0, 1))

            # Prepare the image for Foolbox
            images_tensor = tf.convert_to_tensor(img_expanded, dtype=tf.float32)
            labels_tensor = tf.constant([original_class_id])

            # Generate adversarial example
            attack = attacks.FGSM()
            adversarial_examples = attack(fmodel, images_tensor, labels_tensor)

            # Check if adversarial example was generated
            if adversarial_examples is None:
                logging.error("Adversarial attack failed to generate an example.")
                return render_template(
                    "adversarial.html", message="Adversarial attack failed."
                )

            # Convert back to numpy
            adv_img = adversarial_examples.numpy()[0]

            # Predict the adversarial class
            adv_prediction = model.predict(np.expand_dims(adv_img, axis=0))
            adv_class_id = np.argmax(adv_prediction)
            adv_class_name = sign_names.get(str(adv_class_id), "Unknown")

            # Save adversarial image
            adv_img_path = os.path.join(app.config["UPLOAD_FOLDER"], "adv_" + filename)
            adv_img_uint8 = (adv_img * 255).astype(np.uint8)
            cv2.imwrite(adv_img_path, adv_img_uint8)

            return render_template(
                "adversarial.html",
                original_class_id=original_class_id,
                original_class_name=original_class_name,
                adv_class_id=adv_class_id,
                adv_class_name=adv_class_name,
                original_image_path=filepath,
                adversarial_image_path=adv_img_path,
            )
    else:
        return render_template("adversarial.html")


if __name__ == "__main__":
    app.run(debug=True)
