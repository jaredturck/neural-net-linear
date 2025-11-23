
class Datasets:

    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz '
        self.train_x = []
        self.train_y = []
        self.labels = [
            'amphibian', 'arachnid', 'bird', 'cnidarian', 'crustacean', 'echinoderm', 
            'fish', 'insect', 'mammal', 'marsupial', 'mollusk', 'reptile'
        ]
    
    def argmax(self, x_array):
        ''' Return index of max value in array '''
        max_value = 0
        max_index = 0
        for n,x in enumerate(x_array):
            if x > max_value:
                max_value = x
                max_index = n
        return max_index

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
                    x = x + ([0] * (18 - len(x)))
                    y = [0] * 12
                    y[self.labels.index(family)] = 1
                    self.train_x.append(x)
                    self.train_y.append(y)
