from constants import *

import random
import torch

class FH_Pattern:
    def __init__(self, L = NUM_HOPS_PER_PATTERN, device = "cpu"):
        self.L = L
        self.device = device

        # self.seeds = [random.randint(0, 1000) for i in range(NUM_SEEDS)]
        self.seeds = [616, 52, 218]
        self.prng_objects = [random.Random(seed) for seed in self.seeds]

    def generate_sequence(self, seed_index):
        prng_object = self.prng_objects[seed_index]
        return torch.tensor([prng_object.randint(0, NUM_CHANNELS-1) for i in range(self.L)], device = self.device)

    # def get_pattern(self, index):
        # return self.patterns[index]
    
    # def print_patterns(self):
    #     for i in range(self.M):
    #         print("Pattern ", i, ": ", self.patterns[i])

if __name__ == '__main__':
    fh = FH_Pattern()
    print("Seeds: ", fh.seeds)
    print("Seed 1: ", fh.generate_sequence(0))
    print("Seed 2: ", fh.generate_sequence(1))
    print("Seed 3: ", fh.generate_sequence(2))