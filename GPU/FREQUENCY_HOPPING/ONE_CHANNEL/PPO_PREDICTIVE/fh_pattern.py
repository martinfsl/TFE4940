from constants import *

import random
import torch

class FH_Pattern:
    def __init__(self, M = NUM_PATTERNS, L = NUM_CHANNELS_PER_PATTERN, d = MINIMUM_SPACING, seed = SEED, device = "cpu"):
        self.M = M
        self.L = L
        self.d = d
        self.device = device
        self.patterns = torch.zeros(M, L, device = self.device)
        self.seed = seed
        self.generate_patterns()

    def generate_patterns(self):
        random.seed(self.seed)

        # for i in range(self.M):
        #     used_channels = set()
        #     for j in range(self.L):
        #         while True:
        #             channel = random.randint(0, NUM_CHANNELS - 1)
        #             if channel in used_channels:
        #                 continue
        #             if j == 0 or abs(channel - int(self.patterns[i][j-1].item())) >= self.d:
        #                 self.patterns[i][j] = channel
        #                 used_channels.add(channel)
        #                 break

        pattern_used = [set() for _ in range(self.M)]
        for j in range(self.L):
            used_at_time_step = set()
            for i in range(self.M):
                available = set(range(NUM_CHANNELS)) - pattern_used[i] - used_at_time_step
                valid = [ch for ch in available if (j == 0 or (abs(ch - int(self.patterns[i][j-1].item())) >= self.d))]
                if not valid:
                    raise ValueError(f"No valid channels available for pattern {i}, time step {j}")
                channel = random.choice(valid)
                self.patterns[i][j] = channel
                pattern_used[i].add(channel)
                used_at_time_step.add(channel)

    def get_pattern(self, index):
        return self.patterns[index]
    
    def print_patterns(self):
        for i in range(self.M):
            print("Pattern ", i, ": ", self.patterns[i])

if __name__ == '__main__':
    fh = FH_Pattern()
    fh.print_patterns()
    fh.get_pattern(0)
    fh.get_pattern(1)