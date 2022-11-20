from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


img_width = 320
img_height = 240

mdl = Sequential(
    [layers.Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding='same', input_shape=(img_width, img_height, 3)),
     layers.BatchNormalization(),
     layers.Activation('relu'),
     layers.MaxPooling2D(pool_size=3, strides=2),
     ]
)

print(mdl.summary())
