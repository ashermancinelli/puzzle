#!/usr/bin/env python3
"""
CUDA implementation for exhaustive testing of the chess puzzle XOR solution.

Based on the ProofByContradiction algorithm, this program:
1. Generates all possible NxN board configurations
2. Tests each board with every possible prize location
3. Verifies the XOR solution works correctly using GPU parallelization
"""

import argparse
import time
import math
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def xor_sum_cuda(board_state, board_size):
    """
    Calculate XOR sum (parity) of a board state.
    For each position i where there's a coin (bit set), XOR i into the result.
    """
    result = 0
    for i in range(board_size):
        if (board_state >> i) & 1:
            result ^= i
    return result

@cuda.jit(device=True)
def apply_xor_solution(board_state, prize_index, board_size):
    """
    Apply the XOR solution algorithm:
    1. Calculate original parity
    2. Determine flip position = original_parity ^ prize_index  
    3. Flip that bit
    4. Calculate new parity
    5. Return (flip_position, new_board, new_parity, success)
    """
    original_parity = xor_sum_cuda(board_state, board_size)
    flip_index = original_parity ^ prize_index
    
    # Check bounds
    if flip_index >= board_size:
        return flip_index, board_state, original_parity, False
    
    # Flip the bit at flip_index
    new_board = board_state ^ (1 << flip_index)
    
    # Calculate new parity
    new_parity = xor_sum_cuda(new_board, board_size)
    
    # Success if new parity equals prize index
    success = (new_parity == prize_index)
    
    return flip_index, new_board, new_parity, success

@cuda.jit
def test_all_combinations_kernel(results, board_size, total_combinations):
    """
    CUDA kernel to test all board/prize combinations.
    Each thread handles one combination.
    """
    idx = cuda.grid(1)
    if idx >= total_combinations:
        return
    
    max_board_value = (1 << board_size) - 1
    num_boards = 1 << board_size
    
    # Extract board_index and prize_index from linear index
    board_index = idx // board_size
    prize_index = idx % board_size
    
    if board_index > max_board_value:
        return
    
    # Apply the XOR solution
    flip_pos, new_board, new_parity, success = apply_xor_solution(
        board_index, prize_index, board_size
    )
    
    # Store results: [board, prize, flip_pos, new_board, new_parity, success]
    base_idx = idx * 6
    results[base_idx + 0] = board_index
    results[base_idx + 1] = prize_index  
    results[base_idx + 2] = flip_pos
    results[base_idx + 3] = new_board
    results[base_idx + 4] = new_parity
    results[base_idx + 5] = 1 if success else 0

@cuda.jit
def verify_solution_kernel(results, board_size, total_combinations):
    """
    Secondary verification kernel to double-check results.
    """
    idx = cuda.grid(1)
    if idx >= total_combinations:
        return
    
    base_idx = idx * 6
    board_state = results[base_idx + 0]
    prize_index = results[base_idx + 1]
    flip_pos = results[base_idx + 2] 
    new_board = results[base_idx + 3]
    new_parity = results[base_idx + 4]
    claimed_success = results[base_idx + 5]
    
    # Re-verify the solution
    original_parity = xor_sum_cuda(board_state, board_size)
    expected_flip = original_parity ^ prize_index
    expected_new_board = board_state ^ (1 << expected_flip)
    expected_new_parity = xor_sum_cuda(expected_new_board, board_size)
    
    actual_success = (
        flip_pos == expected_flip and
        new_board == expected_new_board and
        new_parity == expected_new_parity and
        new_parity == prize_index and
        expected_flip < board_size
    )
    
    # Update success flag with verification result
    results[base_idx + 5] = 1 if actual_success else 0

def print_board_visual(board_state, board_size):
    """Print a visual representation of the board."""
    width = int(math.sqrt(board_size))
    if width * width != board_size:
        print(f"Board {board_state:0{board_size}b}")
        return
        
    print(f"Board (decimal {board_state}):")
    for row in range(width):
        row_str = ""
        for col in range(width):
            pos = row * width + col
            bit = (board_state >> pos) & 1
            row_str += "H " if bit else "T "
        print(f"  {row_str}")

def run_exhaustive_test(board_size, verbose=False, verify=True):
    """
    Run exhaustive test of XOR solution on all possible boards and prize locations.
    """
    if board_size > 20:
        print(f"Warning: Board size {board_size} will generate {1 << board_size} boards!")
        print("This may use excessive memory and time.")
        
    num_boards = 1 << board_size
    total_combinations = num_boards * board_size
    
    print(f"Testing board size: {board_size}")
    print(f"Number of boards: {num_boards:,}")
    print(f"Prize locations per board: {board_size}")
    print(f"Total combinations: {total_combinations:,}")
    print(f"Memory required: ~{total_combinations * 6 * 8 / 1024**2:.1f} MB")
    
    # Allocate GPU memory for results
    # Each combination stores: [board, prize, flip_pos, new_board, new_parity, success]
    results_gpu = cuda.device_array(total_combinations * 6, dtype=np.int64)
    
    # Configure CUDA grid
    threads_per_block = 256
    blocks_per_grid = (total_combinations + threads_per_block - 1) // threads_per_block
    
    print(f"CUDA configuration: {blocks_per_grid} blocks × {threads_per_block} threads")
    
    # Run the test kernel
    start_time = time.time()
    test_all_combinations_kernel[blocks_per_grid, threads_per_block](
        results_gpu, board_size, total_combinations
    )
    cuda.synchronize()
    kernel_time = time.time() - start_time
    
    print(f"Kernel execution time: {kernel_time:.3f} seconds")
    print(f"Combinations per second: {total_combinations/kernel_time:,.0f}")
    
    # Optional verification pass
    if verify:
        print("Running verification pass...")
        verify_start = time.time()
        verify_solution_kernel[blocks_per_grid, threads_per_block](
            results_gpu, board_size, total_combinations
        )
        cuda.synchronize()
        verify_time = time.time() - verify_start
        print(f"Verification time: {verify_time:.3f} seconds")
    
    # Copy results back to CPU
    results_cpu = results_gpu.copy_to_host()
    
    # Analyze results
    success_count = 0
    failure_examples = []
    
    for i in range(total_combinations):
        base_idx = i * 6
        success = results_cpu[base_idx + 5]
        if success:
            success_count += 1
        else:
            if len(failure_examples) < 5:  # Collect first few failures
                failure_examples.append({
                    'board': results_cpu[base_idx + 0],
                    'prize': results_cpu[base_idx + 1],
                    'flip_pos': results_cpu[base_idx + 2],
                    'new_board': results_cpu[base_idx + 3],
                    'new_parity': results_cpu[base_idx + 4]
                })
    
    print(f"\nResults:")
    print(f"Successful combinations: {success_count:,}/{total_combinations:,}")
    print(f"Success rate: {100*success_count/total_combinations:.1f}%")
    
    if failure_examples:
        print(f"\nFirst {len(failure_examples)} failure examples:")
        for i, failure in enumerate(failure_examples):
            print(f"  Failure {i+1}:")
            print(f"    Board: {failure['board']}")
            print(f"    Prize: {failure['prize']}")  
            print(f"    Flip position: {failure['flip_pos']}")
            print(f"    New parity: {failure['new_parity']}")
            if verbose:
                print_board_visual(failure['board'], board_size)
    
    # Show some successful examples if verbose
    if verbose and success_count > 0:
        print(f"\nShowing first few successful examples:")
        shown = 0
        for i in range(total_combinations):
            base_idx = i * 6
            if results_cpu[base_idx + 5] and shown < 3:
                board = results_cpu[base_idx + 0]
                prize = results_cpu[base_idx + 1]
                flip_pos = results_cpu[base_idx + 2]
                new_board = results_cpu[base_idx + 3]
                
                print(f"\nExample {shown + 1}:")
                print(f"  Prize at position: {prize}")
                print(f"  Flip position: {flip_pos}")
                print_board_visual(board, board_size)
                print("  After flip:")
                print_board_visual(new_board, board_size)
                shown += 1
    
    return success_count == total_combinations

def benchmark_performance(max_board_size=10):
    """Benchmark performance across different board sizes."""
    print("Performance Benchmark:")
    print("Board Size | Combinations | Time (s) | Comb/sec")
    print("-" * 50)
    
    for size in range(4, max_board_size + 1):
        if size > 16:
            continue  # Skip very large sizes for benchmarking
            
        num_combinations = (1 << size) * size
        if num_combinations > 10**7:  # Skip if > 10M combinations
            print(f"    {size:2d}     | {'> 10M':>11} | {'skipped':>8} | {'--':>8}")
            continue
            
        start_time = time.time()
        run_exhaustive_test(size, verbose=False, verify=False)
        elapsed = time.time() - start_time
        
        rate = num_combinations / elapsed if elapsed > 0 else 0
        print(f"    {size:2d}     | {num_combinations:>11,} | {elapsed:>8.3f} | {rate:>8.0f}")

def main():
    parser = argparse.ArgumentParser(description='CUDA exhaustive test of chess puzzle XOR solution')
    parser.add_argument('-s', '--board-size', type=int, default=4,
                       help='Board size (number of cells, must be perfect square for visualization)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output and examples')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification pass (faster)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark across multiple sizes')
    parser.add_argument('--max-benchmark-size', type=int, default=12,
                       help='Maximum board size for benchmarking')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_performance(args.max_benchmark_size)
        return
    
    # Check CUDA availability
    if not cuda.is_available():
        print("Error: CUDA is not available!")
        return
    
    print(f"CUDA device: {cuda.get_current_device().name}")
    print(f"CUDA compute capability: {cuda.get_current_device().compute_capability}")
    
    success = run_exhaustive_test(
        args.board_size, 
        verbose=args.verbose,
        verify=not args.no_verify
    )
    
    if success:
        print("\n✅ All tests PASSED! The XOR solution works for all possible boards and prize locations.")
    else:
        print("\n❌ Some tests FAILED! There may be an issue with the XOR solution or implementation.")

if __name__ == '__main__':
    main() 
