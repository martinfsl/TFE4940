from constants import *

import random
import torch

class FH_Pattern:
    def __init__(self, L = NUM_HOPS, device = "cpu"):
        self.L = L
        self.device = device

        # Initialize PRNG objects
        random.seed(100)
        self.seeds = [random.randint(0, 1000) for _ in range(NUM_SEEDS)]
        # self.seeds = [616, 52, 218]
        self.prng_objects = [random.Random(seed) for seed in self.seeds]

        self.sequences = torch.tensor([], device=self.device)

    # Generate a sequence of random numbers from all PRNG objects to keep them in sync,
    # but only return the one corresponding to seed_index.
    def generate_sequence(self):
        self.sequences = torch.tensor([], device=self.device)
        for prng in self.prng_objects:
            seq = [prng.randint(0, NUM_CHANNELS - 1) for _ in range(self.L)]
            self.sequences = torch.cat((self.sequences, torch.tensor(seq, device=self.device).unsqueeze(0)), dim=0)

    def get_sequence(self, seed_index):
        if self.L == 1:
            seed_index = seed_index.unsqueeze(0).long()

        return self.sequences[seed_index].squeeze(0)
    
    def get_seed(self, observed_channels):
        for i, seq in enumerate(self.sequences):
            if (seq == observed_channels).sum().item() > self.L / 2:
                return torch.tensor([i], device=self.device)
        return torch.tensor([-1], device=self.device)

if __name__ == '__main__':
    fh1 = FH_Pattern()
    fh2 = FH_Pattern()
    print("Seeds: ", fh1.seeds)
    print("Time step 1")
    print("fh1, seed 1: ", fh1.generate_sequence(0))
    print("fh2, seed 2: ", fh2.generate_sequence(1))
    print("Time step 2")
    print("fh1, seed 2: ", fh1.generate_sequence(1))
    print("fh2, seed 1: ", fh2.generate_sequence(0))
    print("Time step 3")
    print("fh1, seed 2: ", fh1.generate_sequence(1))
    print("fh2, seed 1: ", fh2.generate_sequence(0))
    print("Time step 4")
    print("fh1, seed 1: ", fh1.generate_sequence(0))
    print("fh2, seed 1: ", fh2.generate_sequence(0))