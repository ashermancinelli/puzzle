
import math
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import cvc5
import z3
from z3 import (
    And,
    BitVec,
    BitVecNumRef,
    BitVecRef,
    BitVecSort,
    BitVecVal,
    Extract,
    ForAll,
    Function,
    Or,
    Solver,
    Sum,
    ZeroExt,
)
print(f'{cvc5.__version__=}')

class Packed(np.ndarray):
    def __new__(cls, input_array):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        obj = np.asarray(input_array, dtype=np.uint8).view(cls)
        return obj
    
    def pack(self) -> np.uint64:
        return np.packbits(self)[0]

    def packed_repr(self, width: int = 8) -> str:
        return np.binary_repr(self.pack(), width=width)

    def z3(self, sort: int = 8) -> BitVecNumRef:
        return BitVecVal(self.pack(), sort)

    @classmethod
    def from_binary_string(cls, s: str) -> 'Packed':
        return cls(np.array([int(c) for c in s], dtype=np.uint8))

def test_packing():
    arr = Packed([0, 1, 1, 1, 0, 0, 0, 1])
    assert arr.pack() == 0b01110001
    assert arr.packed_repr() == '01110001'

def popcount(bv):
    assert isinstance(bv, BitVecRef)
    """Calculates the popcount of a Z3 BitVec using iterative summation."""
    size = bv.size()
    bits = [Extract(i, i, bv) for i in range(size)]
    # ZeroExt ensures each bit is treated as a single-bit BitVec for summation
    return Sum([ZeroExt(size - 1, b) for b in bits])


@dataclass
class Problem:
    board: Packed
    money: int

@dataclass
class Solution:
    board: Packed
    flip_index: int
    flipped_board: Packed
    guess: int

class Chess:
    def __init__(self, width: int):
        self.width = width
        self.s = Solver()
        self.s.set('proof', True)
        self.s.set('proof.save', True)
        self.s.set('sls.enable', True)
        self.s.set('unsat_core', True)
        self.board_bv_sort = BitVecSort(width**2)
        # Smallest power capable of representing every cell
        power = int(math.log2(width**2))
        self.cell_index_sort = self.board_bv_sort
        # self.cell_index_sort = BitVecSort(power)
        print(f'{width=} {power=} {self.cell_index_sort=} {self.board_bv_sort=}')
        self.F = Function('flip_function', self.board_bv_sort, self.board_bv_sort, self.board_bv_sort)
        self.G = Function('guess_function', self.board_bv_sort, self.cell_index_sort)
        a_board = BitVec('a_board', self.board_bv_sort)
        a_money_mask = BitVec('a_money_mask', self.board_bv_sort)

        # Assert that the flip function results in a board that equals a power of
        # two when XORed with the original board (indicating only one bit is flipped)
        self.s.add(
            ForAll(
                [a_board, a_money_mask],
                And([
                    # Declare that the money mask is only a single cell
                    popcount(a_money_mask) == 1,
                    # Ensure only a single cell has been flipped
                    popcount(a_board ^ self.F(a_board, a_money_mask)) == 1,
                ])
            )
        )
        # For all possible input boards and money locations, the flip function
        # should return an index such that the input board with the flip function's
        # output bit flipped, passed to the guess function, should return the money index.
        self.s.add(
            ForAll(
                [a_board, a_money_mask],
                And([
                    # Ensure the guessing function always returns the money mask
                    self.G(self.F(a_board, a_money_mask)) == a_money_mask,
                ])
            )
        )

    def check(self):
        if self.s.check() != z3.sat:
            print(self.s)
            print(f'{self.s.unsat_core()=}')
            print(self.s.sexpr())
            assert False, f"Problem is not satisfiable:\n{self.s}\n{self.s.sexpr()}"

    def add_problem(self, problem: Problem):
        print(f'{problem=} {problem.board.packed_repr()=}')
        zboard = problem.board.z3(self.width**2)
        zmoney = BitVecVal(1 << problem.money, self.cell_index_sort)
        self.s.add(
            self.G(
                self.F(zboard, zmoney)
            )
            == zmoney
        )
        self.check()

    def solve(self, problem: Problem):
        print(f'{problem=}')
        self.check()
        m = self.s.model()
        flip = m.evaluate(
            self.F(problem.board.z3(self.width**2), BitVecVal(1 << problem.money, self.board_bv_sort))
        )
        print(f'{flip=}')
        assert isinstance(flip, BitVecNumRef)
        flipped_board = m.evaluate(problem.board.z3(self.width**2) ^ flip)
        assert isinstance(flipped_board, BitVecNumRef)
        print(f'{flipped_board.as_binary_string()=}')
        guess = m.evaluate(self.G(flipped_board))
        assert isinstance(guess, BitVecNumRef)
        print(f'{guess=}')
        return Solution(
            board=problem.board,
            flip_index=flip.as_long(),
            flipped_board=Packed.from_binary_string(flipped_board.as_binary_string()),
            guess=guess.as_long(),
        )


def test_solver_2x2():
    c = Chess(2)
    p = Problem(Packed([0, 1, 1, 0]), 2)
    c.add_problem(p)
    sol = c.solve(p)
    print(f'{sol=}')

if __name__ == '__main__':
    test_solver_2x2()
