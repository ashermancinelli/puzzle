from dataclasses import dataclass
from math import gcd
from typing import Generator

@dataclass
class SetList:
    sets: list[set]

    def insert_pair(self, fst: int, snd: int):
        set_idx_fst = self.find(fst)
        set_idx_snd = self.find(snd)
        match set_idx_fst, set_idx_snd:
            case None, None:
                self.sets.append({fst, snd})
            case (only_one, None):
                self.sets[only_one].add(snd)
            case (None, only_one):
                self.sets[only_one].add(fst)
            case left, right if left != right:
                right_set = self.sets[right]
                for e in right_set:
                    self.sets[left].add(e)
                self.sets.pop(right)

    def find(self, needle: int) -> int | None:
        for index, a_set in enumerate(self.sets):
            if needle in a_set:
                return index
        return None

class Solution:

    @staticmethod
    def prime_factors(n: int) -> list[int]:
        all_factors = list(range(n))
        for i in range(2, n):
            if all_factors[i] < i: # we've already found something smaller
                continue
            # otherwise, nobody has found a smaller factor. Walk from the square
            # up to the max number by steps of i to find new factors.
            for candidate_factor in range(i ** 2, n, i):
                all_factors[candidate_factor] = min(all_factors[candidate_factor], i)

        return all_factors

    @staticmethod
    def all_factors_for_num(n: int, all_factors: list[int]) -> Generator[int, None, None]:
        while n > 1:
            factor = all_factors[n]
            yield factor
            n //= factor


    def gcdSort(self, nums: list[int]) -> bool:
        all_factors = self.prime_factors(max(nums) + 1)
        print(all_factors, len(all_factors))
        setlist = SetList([])
        for num in nums:
            for factor in self.all_factors_for_num(num, all_factors):
                setlist.insert_pair(num, factor)
        sorted_nums = sorted(nums)
        return all(
            [
                setlist.find(num) == setlist.find(sorted_num)
                for num, sorted_num in zip(nums, sorted_nums)
            ]
        )
        
def test():
    s = Solution()
    assert s.gcdSort([7,21,3])
    assert s.gcdSort([10,5,9,3,15])
