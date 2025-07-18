import numpy as np
import numpy.typing as npt
from functools import cache

Arr = np.flatiter
print=lambda *a,**kw:None

class Solution:
    def __init__(self) -> None:
        self.nph: npt.NDArray[np.int32] | None = None
        
    def maxArea(self, hs: list[int]) -> int:
        l, r, m = 0, len(hs)-1, 0
        while l < r:
            w = r-l
            a,b=hs[l],hs[r]
            h = min(a,b)
            area = w*h
            m=max(m,area)
            l, r = (l+1, r) if a<b else (l, r-1)
        return m

def test():
    s=Solution()

    assert s.maxArea([1,8,6,2,5,4,8,3,7]) == 49

    import t1
    assert s.maxArea(t1.test) == t1.answer
