from neuralnet import (
    DataLoader,
    Dataset,
    F_RELU,
    F_SOFTMAX,
    LABEL,
    Model,
    RandomSampler,
    TEXT,
    train,
)


class AnimalModel(Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_linear(input_size, 16, F_RELU)
        self.add_linear(16, output_size, F_SOFTMAX)


if __name__ == '__main__':
    dataset = Dataset().csv(
        'train.txt',
        x=[0],
        y=[1],
        types={0: TEXT, 1: LABEL},
    )

    sampler = RandomSampler(dataset, seed=42)
    loader = DataLoader(dataset, sampler, batch_size=16)
    model = AnimalModel(loader.input_size, loader.output_size)

    print('[+] Training started')
    train(model, loader, 5000, 0.001)

    for animal in ['cat', 'spider', 'salmon']:
        prediction = model.forward(dataset.encode(animal))
        print(f'{animal}: {dataset.label(prediction)}')

    model.free()
    loader.free()
    sampler.free()
    dataset.free()
