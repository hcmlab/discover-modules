import os
os.environ['KERAS_BACKEND'] = 'torch'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''  # cpu only
import keras as keras

class MultitaskMobileNetV2Model:
    def __init__(
        self,
        weights="imagenet",
        input_width=224,
        input_height=224,
        input_channels=3,
        emotion_classes=8,
        emotion_dimensions=2,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.emotion_classes = emotion_classes
        self.emotion_dimensions = emotion_dimensions  # valence/arousal
        self.output_shape = {
            "valence_arousal": (self.emotion_dimensions),
            "emotions": (self.emotion_classes),
        }

        # Create the base pre-trained model
        i = keras.layers.Input(shape=[self.input_width, self.input_height, self.input_channels])#tf.uint8)
        x = keras.applications.mobilenet.preprocess_input(i)

        base_model = keras.applications.MobileNetV2(
            input_tensor=x,
            weights=None,
            include_top=False,
        )

        # Add model head 1
        y0 = keras.layers.GlobalAveragePooling2D()(base_model.output)
        y0 = keras.layers.Dense(
            self.emotion_dimensions, activation="tanh", name="valence_arousal"
        )(y0)

        # Add Model head 2
        y1 = keras.layers.GlobalAveragePooling2D()(base_model.output)
        y1 = keras.layers.Dense(
            self.emotion_classes, activation="softmax", name="emotions"
        )(y1)

        self.model = keras.Model(
            inputs=[i], outputs=[y0, y1], name="MobileNetV2_Multitask"
        )

        # no grad
        for layer in self.model.layers:
            layer.trainable = False