from dataclasses import dataclass
from functools import cache
from typing import Union, Callable

print = lambda *a, **kw: None
PatType = Union['Dot', 'Char', 'Star', 'End']

@dataclass
class Dot:
    cont: PatType

    def __repr__(self): return f'Dot({self.cont})'
    def __hash__(self): return hash((self.cont))
    
@dataclass
class Char:
    c: str
    cont: PatType
    def __repr__(self): return f'Char({self.c},{self.cont})'
    def __hash__(self): return hash((self.c, self.cont))
    
@dataclass
class Star:
    c: str
    cont: PatType
    def __repr__(self): return f'Star({self.c},{self.cont})'
    def __hash__(self): return hash((self.c, self.cont))

class End:
    def __repr__(self): return 'End()'
    def __hash__(self): return hash(None)

@cache
def run(s: str, pat: PatType):
    print(f"'{s}'", pat)
    n = s[1:] if len(s) else ''
    match s, pat:
        case '', End():
            print('good end')
            return True
        case _, End():
            print('incomplete match')
            return False
        case '', Star(_, cont):
            return run('', cont)
        case '', _:
            print('leftover patterns')
            return False
        case c1, Star(c, cont) as star:
            if c == '.' or c1[0] == c:
                print(f'try this star again on {n}')
                if run(n, star):
                    return True
                # print(f'this star again failed, try cont on {n}')
                # if run(n, cont):
                #     return True
                print('this star failed and its cont failed, falling back on cont with og str')
            print('star doesn\'t match, skip it')
            return run(s, cont)  # Skip the star entirely, don't consume characters
        case c1, Char(c, cont):
            return c1[0] == c and run(n, cont)
        case c1, Dot(cont):
            return run(n, cont)
        case _:
            assert False, 'non exhaustive?'

class Solution:
    def compile(self, pat):
        global end
        conts: list[PatType] = []
        i = 0
        e = End()
        while i < len(pat):
            c = pat[i]
            isstar = i + 1 < len(pat) and pat[i+1] == '*'
            if isstar:
                conts.append(Star(c, e))
                i += 1
            elif c == '.':
                conts.append(Dot(e))
            else:
                conts.append(Char(c, e))
            i += 1

        conts.append(e)

        for i, c in enumerate(conts):
            if isinstance(c, End):
                break
            c.cont = conts[i+1]

        print(conts[0])
        return conts[0]

    def isMatch(self, s: str, p: str) -> bool:
        print('-' * 72)
        print(f'isMatch({s}, {p})')
        re: PatType = self.compile(p)
        return run(s, re)

def test_me():
    s = Solution()
    assert s.isMatch('aa', 'a*')
    assert not s.isMatch('aa', 'a')
    assert s.isMatch('ab', '.*')
    assert s.isMatch('aab', 'c*a*b')
    assert not s.isMatch('aaaaaaaaaaaaaab', 'a*a*a*a*a*a*a*a*')
    assert not s.isMatch('aaaaaaaaaaaaaaaaaaab', 'a*a*a*a*a*a*a*a*a*a*')
    assert s.isMatch('aaa', 'a*a')
    assert s.isMatch('bbbba', '.*a*a')
