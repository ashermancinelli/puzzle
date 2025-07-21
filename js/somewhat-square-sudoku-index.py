#!/usr/bin/env -S uv run --script --with z3-solver --with numpy

# https://www.janestreet.com/puzzles/somewhat-square-sudoku-index/

'''
Fill the empty cells in the grid above with digits such that each row, column, and outlined 3-by-3 box contains the same set of nine unique digits1, and such that the nine 9-digit numbers2 formed by the rows of the grid has the highest-possible GCD over any such grid.

Some of the cells have already been filled in. The answer to this puzzle is the 9-digit number formed by the middle row in the completed grid.

that is, you'll be using nine of the ten digits (0-9) in completing this grid ↩

possibly with a leading 0 ↩
'''

import math
import z3
import os
import numpy as np
from timeit import timeit

puzzle = np.array([
    [-1, -1, -1, -1, -1, -1, -1,  2, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1,  5],
    [-1,  2, -1, -1, -1, -1, -1, -1, -1],

    [-1, -1,  0, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1,  2, -1, -1, -1, -1, -1, -1],

    [-1, -1, -1, -1,  0, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1,  2, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1,  5, -1, -1],
])
groups = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 2, 2, 2],

    [3, 3, 3, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 5, 5, 5],

    [6, 6, 6, 7, 7, 7, 8, 8, 8],
    [6, 6, 6, 7, 7, 7, 8, 8, 8],
    [6, 6, 6, 7, 7, 7, 8, 8, 8],
])

def main():
    s = z3.Solver()
    
    # The digit that won't be used in the puzzle (one of 0-9)
    not_in_board = z3.Int('not_in_board')
    s.add(z3.And(not_in_board >= 0, not_in_board <= 9))
    
    # Create board variables
    board = [[z3.Int(f'b_{i}_{j}') for j in range(9)] for i in range(9)]
    threads = int(os.getenv('THREADS', 16))
    print(f'Using {threads=}')
    s.set('smt.threads', threads)

    # Each cell must be a digit 0-9
    for i in range(9):
        for j in range(9):
            s.add(z3.And(board[i][j] >= 0, board[i][j] <= 9))

    # Add predefined values
    for i in range(9):
        for j in range(9):
            if puzzle[i, j] != -1:
                s.add(board[i][j] == puzzle[i, j])

    # Each cell must not be the excluded digit
    for i in range(9):
        for j in range(9):
            s.add(board[i][j] != not_in_board)

    # Row constraints: each row must have distinct values
    for i in range(9):
        s.add(z3.Distinct([board[i][j] for j in range(9)]))

    # Column constraints: each column must have distinct values  
    for j in range(9):
        s.add(z3.Distinct([board[i][j] for i in range(9)]))
    
    # 3x3 box constraints: each box must have distinct values
    for box_id in range(9):
        box_cells = []
        for i in range(9):
            for j in range(9):
                if groups[i, j] == box_id:
                    box_cells.append(board[i][j])
        s.add(z3.Distinct(box_cells))

    solution_count = 0
    best_gcd = 0
    while s.check() == z3.sat:
        solution_count += 1
        print(f'{solution_count=}')

        model = s.model()

        # Extract solution
        evaluated = np.zeros(puzzle.shape, np.int32)
        fresh_constraints = []
        for i in range(9):
            for j in range(9):
                val = model.evaluate(board[i][j])
                evaluated[i, j] = int(str(val))
                fresh_constraints.append(board[i][j] != val)

        print(evaluated)
        print(f"Excluded digit: {model.evaluate(not_in_board)}")
        answer = ''.join(str(evaluated[4, j]) for j in range(9))
        print(f"\nMiddle row (answer): {answer}")
        ianswer = int(answer)
        answer_gcd = math.gcd(ianswer)
        best_gcd = max(best_gcd, answer_gcd)
        print(f'{answer_gcd=} {best_gcd=}')
        if answer == '283950617':
            print('Success')
            raise SystemExit

        s.add(z3.Or(fresh_constraints))

if __name__ == '__main__':
    main()
    # print(timeit(lambda: main()))

