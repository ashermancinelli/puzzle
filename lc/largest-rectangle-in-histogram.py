import numpy as np
from collections import namedtuple
import functools
import inspect

Acc = namedtuple('Acc', 'heigh width'.split())

def ufunc(f):
    """
    Returns the positional and keyword argument names of a function f.
    """
    sig = inspect.signature(f)
    pos_args = []
    kw_args = []
    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            pos_args.append(name)
        elif param.kind == param.KEYWORD_ONLY:
            kw_args.append(name)
    return pos_args, kw_args

def area(acc: Acc, curr: int):
    h, w = acc
    h = min(h, curr)
    w += 1
    return h, w

class Solution:
    def largestRectangleArea(self, hs: list[int]) -> int:
        l = len(hs)
        a = np.zeros((l, l))
        for i, e in enumerate(hs):
            for j in range(i, l):
                h = min(hs[i:j])
        return 0
    
def test():
    s=Solution()
    assert s.largestRectangleArea([2,1,5,6,2,3])==10
