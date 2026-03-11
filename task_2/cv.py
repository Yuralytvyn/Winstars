import json
import pathlib
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


DATASET_DIR = Path("data/raw-img")
MODEL_DIR = Path("models/cv")
MODEL_PATH = MODEL_DIR / "animal_classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_HEAD = 5
EPOCHS_FINE = 3

def check_name_of_classes():
    CLASS_MAP = {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "ragno": "spider",
        "scoiattolo": "squirrel",
    }

    for folder in DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue

        if folder.name in CLASS_MAP:
            new_name = CLASS_MAP[folder.name]
            new_path = folder.parent / new_name

            folder.rename(new_path)

            print(f"{folder.name} -> {new_name}")

def load_full_dataset(data_dir):
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = full_ds.class_names
    return full_ds, class_names


def train_test_split(dataset):
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    train_ds = dataset.take(train_size)
    rest = dataset.skip(train_size)
    val_ds = rest.take(val_size)
    test_ds = rest.skip(val_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)
    print(f"Total batches: {dataset_size}, Train: {train_size}, Val: {val_size}, Test: {dataset_size-train_size-val_size}")
    return train_ds, val_ds, test_ds

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE,3), include_top=False, weights='imagenet'
    )
    base.trainable = False
    inputs = keras.Input(shape=(*IMAGE_SIZE,3))
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.base_model = base
    return model

def build_and_train(train_ds, val_ds, num_classes):
    model = build_model(num_classes)

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    model.base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    return model


def predict_image(image_input):
    if isinstance(image_input, (str, Path)):
        img = keras.preprocessing.image.load_img(image_input, target_size=IMAGE_SIZE)
    elif isinstance(image_input, Image.Image):
        img = image_input.resize(IMAGE_SIZE)
    else:
        img = Image.fromarray(image_input).resize(IMAGE_SIZE)

    arr = keras.preprocessing.image.img_to_array(img)
    arr = tf.expand_dims(arr, axis=0)

    model = keras.models.load_model(MODEL_PATH)
    class_names = sorted([folder.name for folder in DATASET_DIR.iterdir() if folder.is_dir()])

    preds = model.predict(arr, verbose=0)
    idx = tf.argmax(preds[0]).numpy()

    return class_names[idx]