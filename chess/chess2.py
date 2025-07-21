#!/usr/bin/env python3

"""
There's a chessboard with a quarter on each square, each showing either heads or tails at random.
A prize is hidden under one random square.

Player 1 enters the room, flips exactly one coin of their choosing, and then leaves.
Player 2 then enters and must choose the square where the prize is hidden.
"""

import pytest
import z3
import argparse
import math
import itertools
import numpy as np
import numpy.typing as npt

class BV:
    '''
    Wrapper around a Z3 bitvector that allows indexing.
    '''
    def __init__(self, bv):
        self.bv = bv
    def __getitem__(self, idx: int) -> bool:
        idx = idx % self.bv.params()[1]
        return bool(self.bv.as_long() & (1 << idx))

class Solution:
    def __init__(self, width: int):
        self.s = s = z3.Solver()
        self.W = W = width
        board_size = W * W
        power = int(math.log2(board_size)) # power of 2 needed to represent a single cell
        assert 2**power == board_size, 'Board size must be a power of 2 (must be square)'

        self.cell_sort = z3.BitVecSort(power)
        self.board_sort = z3.BitVecSort(board_size)
        self.a_board = z3.BitVec('a_board', self.board_sort)
        self.a_cell = z3.BitVec('a_cell', self.cell_sort)

        self.flip = flip = z3.Function('flip', self.board_sort, self.cell_sort, self.board_sort)

        # assert that only a single bit has been flipped by the flip function.
        # The input board xor-ed with the output board must yield a power of two.
        s.add(
            z3.ForAll(
                [self.a_board, self.a_cell],
                z3.Or(
                    [
                        (flip(self.a_board, self.a_cell) ^ self.a_board) == z3.BitVecVal(2**i, board_size)
                        for i in range(board_size)
                    ]
                ),
            )
        )

        # Guess function always returns a number corresopnding to the chess square
        # that the flipper intended to communicate.
        self.guesser = guesser = z3.Function('guesser', self.board_sort, self.cell_sort)
        s.add(
            z3.ForAll(
                [self.a_board, self.a_cell],
                guesser(flip(self.a_board, self.a_cell)) == self.a_cell,
            )
        )

    def board_to_bitvec(self, board: npt.NDArray[np.uint]) -> z3.BitVecRef:
        bv: z3.BitVecRef = z3.BitVecVal(0, self.board_sort)
        for i, b in enumerate(board.flatten()):
            bv |= int(bool(b)) << i
        return bv

    def print_bv(self, bv: z3.BitVecNumRef):
        print(bv.as_binary_string().zfill(bv.size()))


    def add(self, board: npt.NDArray[np.uint], money: int):
        board_constraint = self.board_to_bitvec(board)
        target_constraint = z3.BitVecVal(money, self.cell_sort)
        self.s.add(
            self.guesser(self.flip(board_constraint, target_constraint))
            == target_constraint
        )

    def solve(self, board: npt.NDArray[np.uint]):
        board_value = self.board_to_bitvec(board)
        assert self.s.check() == z3.sat, 'No solution found'
        m = self.s.model()
        guess = m.evaluate(self.guesser(board_value))
        assert isinstance(guess, z3.BitVecNumRef)
        return guess.as_long()

    def solutions(self, board: npt.NDArray[np.uint], money: int):
        assert len(board) == len(board[0]) == self.W, 'Board must be square'

        zboard = self.board_to_bitvec(board)
        ztarget = z3.BitVecVal(money, self.cell_sort)
        z3.set_param('smt.threads', self.threads)

        if self.s.check() == z3.sat:
            m = self.s.model()
            # import pdb; pdb.set_trace()
            zbv = m.evaluate(zboard)
            assert isinstance(zbv, z3.BitVecNumRef)
            print('Original board:')
            self.print_bv(zbv)
            zflipped = m.evaluate(self.flip(zboard, ztarget))
            assert isinstance(zflipped, z3.BitVecNumRef)
            zguess = m.evaluate(self.guesser(zflipped))
            print('Flipped board:')
            self.print_bv(zflipped)
            print('Guess:')
            assert isinstance(zguess, z3.BitVecNumRef)
            print(zguess.as_long())
            print('Target:')
            print(ztarget.as_long())
            assert isinstance(zguess, z3.BitVecNumRef)
            assert zguess.as_long() == ztarget.as_long(), 'Guess does not match target'
            # yield m.evaluate(a_board).as_long(), m.evaluate(a_cell).as_long()
            # print(m)
            # self.s.add(z3.Or(c != m[c] for c in r for r in self.grid))
        else:
            print(self.s.unsat_core())
            assert any, 'No solution found'

        return True

@pytest.mark.parametrize('args', [argparse.Namespace(width=2)])
def test(args: argparse.Namespace):
    W = args.width
    s = Solution(width=W)
    all_boards = list(itertools.product([0, 1], repeat=W*W))
    for idx, flat_board in enumerate(all_boards):
        board = np.array(flat_board, dtype=np.uint).reshape((W, W))
        print(f"Permutation {idx+1}:")
        print(board)
        for money in range(W*W):  # Try all possible 'money' values for a 2x2 board
            print(f"Solving for money = {money}")
            # s2 = Solution(width=2)  # New solver for each case to avoid constraint pollution
            print(s.add(board, money))
            assert s.solve(board) == money, 'Solution does not match target'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use. 1 means no parallelization, 0 means use all available cores.",
    )
    parser.add_argument('--width', type=int, default=4)
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
