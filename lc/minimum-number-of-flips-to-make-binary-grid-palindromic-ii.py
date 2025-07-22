#!/usr/bin/env python3
import z3
from typing import List

def solve(grid: List[List[int]]) -> int | None:
    s = z3.Optimize()
    n = len(grid)
    m = len(grid[0])
    zgrid = [[z3.BitVec(f'grid_{i}_{j}', 1) for j in range(m)] for i in range(n)]
    palindrome_constraints = []
    for i in range(n):
        for j in range(m):
            palindrome_constraints.append(zgrid[i][j] == zgrid[n-i-1][m-j-1])
    s.add(z3.And(palindrome_constraints))
    sum = z3.IntVal(0)
    diff = z3.IntVal(0)
    for i in range(n):
        for j in range(m):
            # s.add(z3.Or(zgrid[i][j] == 0, zgrid[i][j] == 1))
            sum += z3.If(zgrid[i][j] == 1, 1, 0)
            diff += zgrid[i][j] != grid[i][j]
    s.add(sum % 4 == 0)
    objective = s.minimize(diff)
    if s.check() == z3.sat:
        print(s)
        print(objective.value())
        model = s.model()
        value = model.evaluate(diff)
        assert isinstance(value, z3.IntNumRef)
        return int(value.as_long())
    return None

def test():
    assert solve([[1,0,0],[0,1,0],[0,0,1]]) == 3
    assert solve([[0,1],[0,1],[0,0]]) == 2
    assert solve([[1],[1]]) == 2
