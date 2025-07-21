#!/usr/bin/env python3

"""
There's a chessboard with a quarter on each square, each showing either heads or tails at random.
A prize is hidden under one random square.

Player 1 enters the room, flips exactly one coin of their choosing, and then leaves.
Player 2 then enters and must choose the square where the prize is hidden.
"""

import z3
import math
import itertools
import numpy as np
import numpy.typing as npt

class Solution:
    def __init__(self, width: int):
        self.s = s = z3.Solver()
        self.W = W = width
        board_size = W * W
        power = int(math.log2(board_size)) # power of 2 needed to represent a single cell
        assert 2**power == board_size, 'Board size must be a power of 2 (must be square)'

        self.cell_sort = z3.BitVecSort(power)
        board_vars = [z3.BoolSort()] * board_size
        self.board_sort, self.mk_board, self.board_accessors = z3.TupleSort('board', *board_vars)

        self.flipper = z3.Function('flip', self.board_sort, self.cell_sort, self.board_sort)
        self.guesser = z3.Function('guesser', self.board_sort, self.cell_sort)

        a_board = self.mk_board(*[z3.Bool('a_board_%d' % i) for i in range(board_size)])
        a_cell = z3.BitVec('a_cell', self.cell_sort)

        # assert that only a single bit has been flipped by the flip function.
        # The input board xor-ed with the output board must yield a power of two.
        
        s.add(
            z3.ForAll(
                [a_board, a_cell],
                z3.Or(
                    [
                        (self.flipper(a_board, a_cell) ^ a_board) == z3.BitVecVal(2**i, board_size)
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


    def solutions(self, board: npt.NDArray[np.uint], money: int):
        assert len(board) == len(board[0]) == self.W, 'Board must be square'

        zboard = self.board_to_bitvec(board)
        ztarget = z3.BitVecVal(money, self.cell_sort)
        self.s.add(self.guesser(self.flip(zboard, ztarget)) == ztarget)

        any = False
        if self.s.check() == z3.sat:
            any = True
            m = self.s.model()
            print('Original board:\n', zboard, zboard.sort())
            zflipped = m.evaluate(self.flip(zboard, ztarget))
            zguess = m.evaluate(self.guesser(zflipped))
            print('Flipped board:\n', zflipped)
            print('Guess:\n', zguess)
            print('Target:\n', ztarget)
            assert isinstance(zguess, z3.BitVecNumRef)
            assert zguess.as_long() == ztarget.as_long(), 'Guess does not match target'
            # yield m.evaluate(a_board).as_long(), m.evaluate(a_cell).as_long()
            # print(m)
            # self.s.add(z3.Or(c != m[c] for c in r for r in self.grid))
        else:
            print(self.s.unsat_core())
            assert any, 'No solution found'

        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of threads to use. 1 means no parallelization, 0 means use all available cores.",
    )
    parser.add_argument('--width', type=int, default=4)
    args = parser.parse_args()

    import os
    num_threads = [
        os.cpu_count() or 1,
        1,
        args.threads
    ][min(args.threads, 2)]
    z3.set_param('smt.threads', num_threads)

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
            print(s.solutions(board, money))

if __name__ == '__main__':
    main()
