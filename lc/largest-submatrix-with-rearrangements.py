from typing import List
import numpy as np
from numpy.typing import NDArray
from itertools import permutations

class Solution:

    def score(self, m: NDArray) -> int:
        print('score:\n', m)
        r, c = m.shape
        r, c = (max(0, r-1), max(0, c-1))
        print(r, c)
        best = int(m.any())
        for i in range(r):
            for j in range(c):
                mm = m[i:i+r, j:j+r]
                print(mm)
                f = mm.flat
                sz = len(f)
                c = np.sum(mm)
                if sz == c:
                    best = max(best, sz)
                    print('best:', best)
        return best

    def largestSubmatrix(self, mm: List[List[int]]) -> int:
        m = np.array(mm)
        cs = list(m[:, i] for i in range(m.shape[1]))
        perms = list(map(np.array, permutations(cs)))
        [print(i) for i in perms]
        best = 0
        for perm in perms:
            best = max(best, self.score(perm))
        print(cs)
        return best

def test():
    s=Solution()
    assert s.largestSubmatrix([[0,0,1],[1,1,1],[1,0,1]])==4
    assert s.largestSubmatrix([[1,0,1,0,1]])==3
