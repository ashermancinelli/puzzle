from itertools import combinations

class Solution:
    def threeSumClosest(self, nums: list[int], target: int) -> int:
        m = sum(nums[:3])
        best = abs(target - m)
        for c in combinations(nums,3):
            s=sum(c)
            diff = abs(target - s)
            if diff < best:
                best=diff
                m=s
        return m
            
    
def test():
    s=Solution()
    assert 0==s.threeSumClosest([0,0,0],1)
    assert 2==s.threeSumClosest([-1,2,1,-4], target = 1)
