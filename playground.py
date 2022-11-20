import tensorflow as tf






input = Input (input_shape)
x = Conv2D(64, 7, strides = 2, padding = 'same')(input)
x = MaxPool2D(3, strides = 2, padding = 'same')(x)