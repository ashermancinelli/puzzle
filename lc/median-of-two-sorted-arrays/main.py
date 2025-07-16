#!/usr/bin/env python
import math
from typing import List
class Solution:
    def lower(self, i, j, a, b):
        return i + 1, a[i] if a[i] < b[j] else j + 1, b[j]

    def findMedianSortedArrays(self, n1: List[int], n2: List[int]) -> float:
        print(n1, n2)
        l1, l2 = len(n1), len(n2)
        medi = (len(n1) + len(n2)) // 2
        even = (l1+l2)%2 == 0
        medi = medi - 1 if even else medi
        i, j, last = 0, 0, 0
        print(l1, l2, medi)

        def getif(idx, list):
            return list[idx] if idx < len(list) else math.inf

        while i + j <= medi:
            a, b = getif(i, n1), getif(j, n2)
            if a <= b:
                print('left')
                last = a
                i += 1
            else:
                print('right')
                last = b
                j += 1

        if even:
            print('even', last)
            return (last + min(getif(i, n1), getif(j, n2))) / 2

        print('odd', last)
        return last

def test_me():
    s = Solution()
    assert 2.5 == s.findMedianSortedArrays([1,2],[3,4])
    assert 2.0 == s.findMedianSortedArrays([1,3],[2])
