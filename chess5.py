#! /usr/bin/env python3
import argparse
import pytest
from collections import namedtuple
from z3 import (
    BitVec,
    BitVecRef,
    BitVecVal,
    Extract,
    If,
    Not,
    Solver,
    ULT,
    ZeroExt,
    unsat,
)

Result = namedtuple('Result', 'proven solver'.split())

class Solution:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.bits = board_size.bit_length()
        self.s = Solver()

    def __call__(self):
        board, prize_index = BitVec('board', self.board_size), BitVec('prize_index', self.bits)
        original_parity = Solution.xor_sum_z3(board, self.board_size)
        flip_index = original_parity ^ prize_index
        self.s.add(
            ULT(prize_index, BitVecVal(self.board_size, self.bits))
        )
        self.s.add(
            ULT(flip_index, BitVecVal(self.board_size, self.bits))
        )
        flip_mask = BitVecVal(1, self.board_size) << ZeroExt(self.board_size - self.bits, flip_index)
        board_prime = board ^ flip_mask
        guess = Solution.xor_sum_z3(board_prime, self.board_size)

        null_hypothesis = Not(guess == prize_index)
        self.s.add(null_hypothesis)
    
        return Result(self.s.check() == unsat, self.s)


    @staticmethod
    def xor_sum_z3(bv: BitVecRef, n: int):
        bits_for_index = n.bit_length()
        acc = BitVecVal(0, bits_for_index)
        for k in range(n):
            bit = Extract(k, k, bv)
            acc = acc ^ If(bit == BitVecVal(1,1),
                        BitVecVal(k, bits_for_index),
                        BitVecVal(0, bits_for_index))
        return acc


@pytest.mark.parametrize('board_size', [2**i for i in (2, 4, 6)])
def test(board_size: int):
    s = Solution(board_size)
    solved, solver = s()
    assert solved

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--board-size', type=int, default=4)
    parser.add_argument(
        "--sexpr",
        action="store_true",
        help="Print the solver in sexpr format, importable by z3 and other SMT solvers.",
    )
    args = parser.parse_args()

    print(f'Config: {args.board_size=}, {args.sexpr=}')
    s = Solution(args.board_size)

    solved, solver = s()
    print(f'{solved=}')
    if args.sexpr:
        print(solver.sexpr())
    else:
        print(f'Failed to solve: {solver.unsat_core()=}')
