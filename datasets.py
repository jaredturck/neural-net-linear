
class Datasets:

    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz '
        self.train = []

    def tokenize(self, x):
        ''' Tokenize string value '''
        return [self.alphabet.index(i) for i in x]

    def animal_families_dataset(self):
        ''' List of animals and their families '''

        with open('train.txt', 'r', encoding='utf-8') as file:
            for row in file:
                if row:
                    animal, family = row.strip().split(',')
                    x = self.tokenize(animal)[:18]
                    y = self.tokenize(family)[:10]

                    # pad
                    x = x + ([0] * (18 - len(x)))
                    y = y + ([0] * (10 - len(y)))
                    self.train.append((x,y))
