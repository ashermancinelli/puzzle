#! /usr/bin/env python3
import argparse
import itertools
import math
import pytest
import numpy as np
import numpy.typing as npt
from collections import namedtuple
from z3 import (
    BitVecNumRef,
    BitVecSort,
    sat,
    BitVec,
    Function,
    ForAll,
    Or,
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

class ProofByCaseAnalysis:
    '''Only possible on small boards.'''
    def __init__(self, width: int):
        self.s = Solver()
        self.W = W = width
        board_size = W * W
        power = int(math.log2(board_size)) # power of 2 needed to represent a single cell
        assert 2**power == board_size, 'Board size must be a power of 2 (must be square)'

        self.cell_sort = BitVecSort(power)
        self.board_sort = BitVecSort(board_size)
        self.a_board = BitVec('a_board', self.board_sort)
        self.a_cell = BitVec('a_cell', self.cell_sort)
        self.threads = 1

        self.flip = Function('flip', self.board_sort, self.cell_sort, self.board_sort)

        # assert that only a single bit has been flipped by the flip function.
        # The input board xor-ed with the output board must yield a power of two.
        self.s.add(
            ForAll(
                [self.a_board, self.a_cell],
                Or(
                    [
                        (self.flip(self.a_board, self.a_cell) ^ self.a_board) == BitVecVal(2**i, board_size)
                        for i in range(board_size)
                    ]
                ),
            )
        )

        # Guess function always returns a number corresopnding to the chess square
        # that the flipper intended to communicate.
        self.guesser = Function('guesser', self.board_sort, self.cell_sort)
        self.s.add(
            ForAll(
                [self.a_board, self.a_cell],
                self.guesser(self.flip(self.a_board, self.a_cell)) == self.a_cell,
            )
        )
        print('-' * 60)
        print('Before examples:')
        print(self.s.sexpr())
        print('-' * 60)

    def board_to_bitvec(self, board: npt.NDArray[np.uint]) -> BitVecRef:
        bv: BitVecRef = BitVecVal(0, self.board_sort)
        for i, b in enumerate(board.flatten()):
            bv |= int(bool(b)) << i
        return bv


    def solutions(self, board: npt.NDArray[np.uint], money: int):
        assert len(board) == len(board[0]) == self.W, 'Board must be square'

        zboard = self.board_to_bitvec(board)
        ztarget = BitVecVal(money, self.cell_sort)
        self.s.add(self.guesser(self.flip(zboard, ztarget)) == ztarget)
        any = False
        if self.s.check() == sat:
            any = True
            m = self.s.model()

            def pbv(bv: BitVecRef) -> str:
                evaled = m.evaluate(bv)
                assert isinstance(evaled, BitVecNumRef)
                return evaled.as_binary_string().zfill(evaled.size())

            zflipped = m.evaluate(self.flip(zboard, ztarget))
            assert isinstance(zflipped, BitVecRef)
            zguess = m.evaluate(self.guesser(zflipped))
            print(pbv(zboard), pbv(zflipped), zguess, ztarget)
            assert isinstance(zguess, BitVecNumRef)
            assert zguess.as_long() == ztarget.as_long(), 'Guess does not match target'
            print(self.s.sexpr())
        else:
            print(self.s.unsat_core())
            assert any, 'No solution found'

        return True

@pytest.mark.parametrize('board_size', [2,])
def test_exhaustive(board_size: int):
    all_boards = list(itertools.product([0, 1], repeat=board_size*board_size))
    print('board flipped target guess')
    for idx, flat_board in enumerate(all_boards):
        board = np.array(flat_board, dtype=np.uint).reshape((board_size, board_size))
        for money in range(board_size*board_size):
            s = ProofByCaseAnalysis(board_size)
            assert s.solutions(board, money)

class ProofByContradiction:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.bits = board_size.bit_length()
        self.s = Solver()

    def __call__(self):
        board, prize_index = BitVec('board', self.board_size), BitVec('prize_index', self.bits)
        original_parity = self.xor_sum_z3(board, self.board_size)
        flip_index = original_parity ^ prize_index
        self.s.add(
            ULT(prize_index, BitVecVal(self.board_size, self.bits))
        )
        self.s.add(
            ULT(flip_index, BitVecVal(self.board_size, self.bits))
        )
        flip_mask = BitVecVal(1, self.board_size) << ZeroExt(self.board_size - self.bits, flip_index)
        board_prime = board ^ flip_mask
        guess = self.xor_sum_z3(board_prime, self.board_size)

        counterexample = Not(guess == prize_index)
        self.s.add(counterexample)
    
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
    s = ProofByContradiction(board_size)
    solved, solver = s()
    assert solved

if __name__ == '__main__':
    test_exhaustive(2)
    raise SystemExit
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--board-size', type=int, default=4)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file for the SMT solution.')
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="sexpr",
        choices=["sexpr", "smt2"],
        help="Output format for the SMT solution.",
    )
    parser.add_argument(
        "--sexpr",
        action="store_true",
        help="Print the solver in sexpr format, importable by z3 and other SMT solvers.",
    )
    args = parser.parse_args()

    print(f'Config: {args.board_size=}, {args.sexpr=}')
    s = ProofByContradiction(args.board_size)

    solved, solver = s()
    print(f'{solved=}')
    if args.output:
        with open(args.output, 'w') as f:
            f.write(solver.sexpr() if args.format == 'sexpr' else solver.to_smt2())
    if args.sexpr:
        print(solver.sexpr())
