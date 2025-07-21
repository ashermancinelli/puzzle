#!/usr/bin/env python3
import tqdm
import numpy as np
from collections import defaultdict
import click
from math import (floor, ceil, sqrt)

jonah = {
    0: [[0,0,0,0], [0,0,1,0], [1,1,0,1], [1,1,1,1]],
    1: [[0,0,0,1], [0,0,1,1], [1,1,0,0], [1,1,1,0]],
    2: [[0,1,1,0], [0,1,1,1], [1,0,0,1], [1,0,0,0]],
    3: [[0,1,0,0], [0,1,0,1], [1,0,1,0], [1,0,1,1]],
}


class Solution:
    def __init__(self, board, prize, show=True, tofile=None):
        self.board = board
        self.prize = prize
        self.show = show
        self._logfile = tofile

    def L(self, *a, **k):
        if self.show:
            if self._logfile is not None:
                end = '\n'
                if 'end' in k:
                    end = k['end']
                    del k['end']
                self._logfile.write(*a, **k)
                self._logfile.write(end)
            else:
                print(*a, **k)

    def solve(self):
        assert False, "NYI"

    def board_int(self):
        i = self.uz(0)
        for e in self.board:
            i <<= 1
            i |= e
        return i

    def F(self, v):
        return format(v, f'0{len(self.board)}b')

    def __str__(self):
        return f"{self.F(self.board_int())}"

class GPTSolution(Solution):
    def parity(self):
        parity = 0
        for i, e in enumerate(self.board):
            if e:
                parity = parity ^ i
        return parity

    def p1(self):
        flip = self.parity()
        flip = (flip ^ self.prize) % len(self.board)
        self.board[flip] = 0 if self.board[flip] else 1
        return flip

    def p2(self):
        p = self.parity()
        return p

    def solve(self, show=False):
        self.L(f'board (prize at {self.prize}):\n{self}')

        flipped = self.p1()

        self.L(f"board (flipped {flipped}):\n{self}")

        guess = self.p2()

        self.L(f"Guess: {guess}")
        assert guess == self.prize

class JonahSolution(Solution):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.map = defaultdict(list)
        self.rmap = dict()
        match len(self.board):
            case 4:
                self.uz = np.uint8
            case 8:
                self.uz = np.uint8
            case 16:
                self.uz = np.uint16
            case 32:
                self.uz = np.uint32
            case 64:
                self.uz = np.uint64
            case _:
                assert False, "Bad board size for this solution"

        self.mask = self.uz(0)
        for i in range(len(self.board)):
            self.mask |= 1 << i

    def C(self, v):
        '''clamp value to int size that maps from board'''
        return self.mask & v

    def convert(self):
        bi = self.board_int()
        l = len(self.board)
        mid = (2 ** l) // 2
        mirror = self.C(((2**l)-1)-bi)
        ret = (mirror if bi >= mid else bi) % l
        self.L(f'midpoint mirror return {self.F(mid)} {self.F(mirror)} {self.F(ret)}')
        return ret

    def flip(self, i):
        self.board[i] = 1 - self.board[i]

    def findflip(self):
        self.L(f'Finding square to flip, board: {self}')
        for i in range(len(self.board)):
            self.flip(i)
            convert = int(self.convert())
            self.L(f'Trying flip {i:04d}: {self.prize=} == {convert=} ({self.F(convert)})')
            if self.prize == convert:
                return i
            self.flip(i) # unflip if it didn't work
        return None

    def p1(self):
        bi = self.board_int()
        self.L(f'board int: {self.F(bi)}')
        flip = self.findflip()
        self.L(f'{flip=}')
        assert flip is not None, "Didnt find flip!!!"
        return flip

    def p2(self):
        return self.convert()

    def solve(self):
        self.L(f'board ({self.prize=}):\n{self}')
        flip = self.p1()
        self.L(f'p1 (flipped square {flip}):\n{self}')
        guess = self.p2()
        self.L(f'p2 guessed {guess}')
        assert guess == self.prize, 'Incorrect answer'

@click.command()
@click.option('--size', default=4)
def main(size):
    assert floor(sqrt(size)) == ceil(sqrt(size)), 'must be square'
    pow2 = pow(2, size)
    with open('log.txt', 'w') as f:
        for i in range(pow2):
            for p in range(size):
                board = list(int(c) for c in format(i, f'0{size}b'))
                b = JonahSolution(board=board, prize=p, show=True, tofile=None)
                # b = GPTSolution(board=board, prize=p)
                b.solve()
                # exit()
            # print(f'{i:020d}/{pow2:020d}', end='\r')

if __name__ == '__main__':
    main()
