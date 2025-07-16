#!/usr/bin/env python

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ''
        
        longest = s[0]  # Initialize with first character
        
        # Check all possible substrings
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):  # j is the end index (exclusive)
                substring = s[i:j]
                # Check if it's a palindrome and longer than current longest
                if substring == substring[::-1] and len(substring) > len(longest):
                    longest = substring
        
        return longest

def test_main():
    s = Solution()
    assert s.longestPalindrome('bb') == 'bb'
    assert s.longestPalindrome('abcdbbfcba') == 'bb'
