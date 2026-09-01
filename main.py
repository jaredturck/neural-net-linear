import neuralnet as nn


class AnimalModel(nn.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_linear(input_size, 16, nn.Activation.RELU)
        self.add_linear(16, output_size, nn.Activation.SOFTMAX)


if __name__ == '__main__':
    dataset = nn.Dataset().csv(
        'train.txt',
        x=[0],
        y=[1],
        types={
            0: nn.FieldType.TEXT,
            1: nn.FieldType.LABEL,
        },
    )

    sampler = nn.RandomSampler(dataset, seed=42)
    loader = nn.DataLoader(dataset, sampler, batch_size=16)
    model = AnimalModel(loader.input_size, loader.output_size)

    print('[+] Training started')
    nn.train(model, loader, 5000, 0.001)

    for animal in ['cat', 'spider', 'salmon']:
        prediction = model.forward(dataset.encode(animal))
        print(f'{animal}: {dataset.label(prediction)}')

    model.free()
    loader.free()
    sampler.free()
    dataset.free()
