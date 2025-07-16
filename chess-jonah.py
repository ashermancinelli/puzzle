import numpy as np
import networkx as nx

class Board:
    def __init__(self, wh=2):
        self.len = 2 ** wh
        self.solution = dict()
        assert self.len % 2 == 0, 'Board size must be even'

    def solve(self):
        pos = 0

        for i in range(self.len):
            self.solution[i] = []
            for j in self.len / 2:
                self.solution[i].append(pos)
                pos += 1

def main():
    pass

if __name__ == '__main__':
    main()
