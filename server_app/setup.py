from os import path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def sepparate_train_test_datasets(dir, image_size, batch_size):
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_ds = train_datagen.flow_from_directory(
        path.join(dir, "train"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_ds = val_datagen.flow_from_directory(
        path.join(dir, "test"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle = False
    )

    val_ds = val_datagen.flow_from_directory(
        path.join(dir, "validation"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_ds, test_ds, val_ds

def setup_model():
    '''
    Loading data into tensors
    '''
    dir = r".\separated_data"
    image_size = (64, 64)
    batch_size = 25
    train_dt, test_dt, val_dt = sepparate_train_test_datasets(
                                                dir, 
                                                image_size, 
                                                batch_size)

    # Model architecture and definition
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=image_size + (3,)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    #Training the model
    callback = EarlyStopping(monitor='loss', patience=3)
    optimizer = Adam(learning_rate=0.0003)

    model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])

    model.fit(
        train_dt,
        steps_per_epoch=len(train_dt),
        epochs=50,
        validation_data=val_dt,
        validation_steps=len(val_dt),
        callbacks=[callback]
    )

    return model

if __name__ == "__main__":
    setup_model()

