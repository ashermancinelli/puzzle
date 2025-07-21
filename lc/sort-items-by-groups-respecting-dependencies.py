from typing import List
import numpy as np
import networkx as nx

class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        ns = np.arange(n)
        mat = np.array([group, ns], dtype=np.int32).T
        print(mat)
        mat = mat[mat[:, 0].argsort()[::-1]]
        maxconstraints = max(map(len, beforeItems))
        consts = np.full((mat.shape[0], maxconstraints), np.nan)
        for i, beforeidx in enumerate(beforeItems):
            for j, b in enumerate(beforeidx):
                consts[i, j] = b
        print(consts)
        print(mat)
        return []

def test():
    s = Solution()
    assert s.sortItems(
        8, 2, [-1, -1, 1, 0, 0, 1, 0, -1], [[], [6], [5], [6], [3, 6], [], [], []]
    ) == [6, 3, 4, 1, 5, 2, 0, 7]
    # assert (
    #     s.sortItems(
    #         8, 2, [-1, -1, 1, 0, 0, 1, 0, -1], [[], [6], [5], [6], [3], [], [4], []]
    #     )
    #     == []
    # )


if __name__ == '__main__':
    test()
