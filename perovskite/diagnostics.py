from keras import backend as K
import numpy as np
import cv2

class Activation:

    def __init__(self, model):

        # Instatiate with a classifier
        self.model = model.model

        # Find the last conv layer
        conv_layers = [layer.name for layer in self.model.layers if 'conv' in layer.name]
        self.last_conv_layer = self.model.get_layer(conv_layers[-1])

    def compute_gradient(self):

        output = self.model.output
        grads = K.gradients(output, self.last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(1, 2))
        return K.function([self.model.input], [pooled_grads, self.last_conv_layer.output[0]])

    def generate_heatmap(self, img_tensor, tol=1e-3):

        iterate = self.compute_gradient()
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

        # if np.sum(pooled_grads_value) == 0.0:
        #     p = len(pooled_grads_value)
        # pooled_grads_value = np.ones(p)
        #
        # pooled_grads_value /= np.sum(pooled_grads_value)
        pooled_grads_value = pooled_grads_value.reshape(pooled_grads_value.size)
        heatmap = np.average(conv_layer_output_value, weights=pooled_grads_value, axis=-1)
        # heatmap = np.mean(conv_layer_output_value, axis=2)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (self.model.input_shape[1], self.model.input_shape[2]))

        return heatmap