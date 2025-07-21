#!/usr/bin/env python3
import numpy as np
import numpy.typing as npt
from typing import Tuple

def xor_sum_numpy(board: npt.NDArray[np.uint]) -> int:
    """Compute the XOR sum (parity) of a board using numpy."""
    flat = board.flatten()
    result = 0
    for i, bit in enumerate(flat):
        if bit:
            result ^= i
    return result

def solve_chess_puzzle(board: npt.NDArray[np.uint], prize_cell: int) -> Tuple[int, npt.NDArray[np.uint]]:
    """
    Solve the chess square puzzle for a given board and prize location.
    
    Args:
        board: 2D numpy array of 0s and 1s representing the chess board
        prize_cell: Integer representing the target cell (0-indexed, row-major order)
    
    Returns:
        Tuple of (flip_cell, new_board) where:
        - flip_cell: The cell index to flip
        - new_board: The resulting board after flipping
    """
    assert board.shape[0] == board.shape[1], "Board must be square"
    board_size = board.shape[0] * board.shape[1]
    assert 0 <= prize_cell < board_size, f"Prize cell must be in range [0, {board_size-1}]"
    
    # Calculate the XOR sum (parity) of the current board
    current_parity = xor_sum_numpy(board)
    
    # The cell to flip is determined by XORing current parity with target
    flip_cell = current_parity ^ prize_cell
    
    # Create the new board by flipping the determined cell
    new_board = board.copy()
    flip_row, flip_col = divmod(flip_cell, board.shape[1])
    new_board[flip_row, flip_col] = 1 - new_board[flip_row, flip_col]
    
    return flip_cell, new_board

def verify_solution(new_board: npt.NDArray[np.uint], expected_prize: int) -> bool:
    """Verify that the solution works by checking if the parity equals the prize cell."""
    computed_parity = xor_sum_numpy(new_board)
    return computed_parity == expected_prize

def print_board_with_indices(board: npt.NDArray[np.uint]):
    """Print a board with cell indices for reference."""
    rows, cols = board.shape
    print("Board with indices:")
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            print(f"{board[r,c]}({idx:2d})", end=" ")
        print()

def demo_puzzle():
    """Demonstrate the chess square puzzle with various examples."""
    print("Chess Square Puzzle Demonstration")
    print("=" * 50)
    
    # Test with a 4x4 board
    print("\n4x4 Board Example:")
    board_4x4 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1], 
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=np.uint)
    
    print("\nOriginal board:")
    print_board_with_indices(board_4x4)
    
    # Test multiple prize locations
    for prize in [5, 10, 15]:
        print(f"\n--- Prize is in cell {prize} ---")
        flip_cell, new_board = solve_chess_puzzle(board_4x4, prize)
        
        print(f"Current board parity: {xor_sum_numpy(board_4x4)}")
        print(f"Target prize cell: {prize}")
        print(f"Cell to flip: {flip_cell}")
        
        print(f"\nBoard after flipping cell {flip_cell}:")
        print_board_with_indices(new_board)
        
        # Verify the solution
        new_parity = xor_sum_numpy(new_board)
        print(f"New board parity: {new_parity}")
        success = verify_solution(new_board, prize)
        print(f"Solution correct: {success} ✓" if success else f"Solution incorrect: {success} ✗")
    
    print("\n" + "=" * 50)
    print("\n2x2 Board Example:")
    board_2x2 = np.array([
        [1, 0],
        [1, 1]
    ], dtype=np.uint)
    
    print("\nOriginal board:")
    print_board_with_indices(board_2x2)
    
    # Test all possible prize locations for 2x2 board
    for prize in range(4):
        print(f"\n--- Prize is in cell {prize} ---")
        flip_cell, new_board = solve_chess_puzzle(board_2x2, prize)
        
        print(f"Cell to flip: {flip_cell}")
        print(f"Board after flipping:")
        print_board_with_indices(new_board)
        
        success = verify_solution(new_board, prize)
        print(f"Solution correct: {success} ✓" if success else f"Solution incorrect: {success} ✗")

if __name__ == "__main__":
    demo_puzzle() 
