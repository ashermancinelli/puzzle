#!/usr/bin/env python
from typing import Callable, NamedTuple, List, Set, Optional, Tuple
import numpy as np
from itertools import product

class Solver:
    def check_substring(self, string: str, start: int) -> int:
        print(f'substr: {start=} {string=}')
        s = set()
        sofar = ''
        for j, c in enumerate(string[start:]):
            if c in s:
                print(f'reset {sofar=}')
                return len(sofar)
            s.add(c)
            sofar += c
        print(f'end {sofar=}')
        return len(sofar)

    def run(self, string: str) -> int:
        longest = 0 if string == '' else 1
        for i, c in enumerate(string):
            print(f'{i=} {c=}')
            sub = self.check_substring(string, i)
            print(f'{sub=}')
            longest = max(longest, sub)
        return longest

    def check(self, input, expect):
        print('-' * 72)
        print(input, expect)
        assert expect == self.run(input)
            

if __name__ == "__main__":
    s = Solver()
    s.check('abcabcbb', 3)
    s.check('aaaaaaaa', 1)
    s.check('pwwkew', 3)
    s.check('au', 2)
    s.check('dvdf', 3)
