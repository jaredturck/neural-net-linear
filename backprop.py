
class Backpropogation:

    @staticmethod
    def compute_softmax_gradients(layer, y):
        ''' Compute graidents for softmax activation '''

        x, z, a = layer.backprop_cache
        deltas = [ai - yi for ai, yi in zip(a,y)]

        layer.gradients = []
        for i in range(layer.output_neurons):
            layer.gradients.append([deltas[i] * xi for xi in x])
        
        layer_deltas = []
        for j in range(layer.input_neurons):
            layer_deltas.append(sum(layer.weights[i][j] * deltas[i] for i in range(layer.output_neurons)))
        
        return layer_deltas
    
    @staticmethod
    def compute_relu_gradients(layer, deltas):
        ''' Compute dragients for relu activation'''

        x, z, a = layer.backprop_cache
        mask = [int(v > 0) for v in z]
        deltas = [d * m for d,m in zip(deltas, mask)]

        layer.gradients = []
        for i in range(layer.output_neurons):
            layer.gradients.append([deltas[i] * xi for xi in x])
        
        layer_deltas = []
        for j in range(layer.input_neurons):
            layer_deltas.append(sum(layer.weights[i][j] * deltas[i] for i in range(layer.output_neurons)))
        
        return layer_deltas
