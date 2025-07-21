#!/usr/bin/env python3
# https://www.janestreet.com/puzzles/somewhat-square-sudoku-index/
import z3
import numpy as np

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
    not_in_board = z3.Int('not_in_board')
    board = [ [ z3.Int(f'b_{i}_{j}') for j in range(9) ] for i in range(9) ]

    for row in board:
        s.add(z3.And([cell != not_in_board for cell in row]))

    group = z3.Function('group', z3.IntSort(), z3.IntSort())
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            s.add(group(cell) == groups[i, j])
            predefined_value = puzzle[i, j]
            if predefined_value != -1:
                s.add(cell == predefined_value)

    cols = [ [ board[i][j] for i in range(9) ] for j in range(9) ]
    for col in cols:
        s.add(z3.And([z3.Distinct(col)]))
    for row in board:
        s.add(z3.And([z3.Distinct(row)]))
    

if __name__ == '__main__':
    main()
