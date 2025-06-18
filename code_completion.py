"""
Task 1: AI-Powered Code Completion
Comparing AI-suggested vs Manual Implementation for Dictionary Sorting
"""

import time
import random
from typing import List, Dict, Any, Union


def manual_implementation_sort_dicts(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    Manual implementation: Sort a list of dictionaries by a specific key
    
    Args:
        data: List of dictionaries to sort
        key: The key to sort by
        reverse: If True, sort in descending order
    
    Returns:
        Sorted list of dictionaries
    
    Time Complexity: O(n log n)
    Space Complexity: O(n) due to creating new list
    """
    # Manual implementation using basic sorting logic
    sorted_data = []
    
    # Create a copy to avoid modifying original
    data_copy = data.copy()
    
    # Manual bubble sort implementation (inefficient but demonstrates manual approach)
    n = len(data_copy)
    for i in range(n):
        for j in range(0, n - i - 1):
            # Compare values at the specified key
            val1 = data_copy[j].get(key, 0)
            val2 = data_copy[j + 1].get(key, 0)
            
            # Swap if needed based on reverse flag
            if (not reverse and val1 > val2) or (reverse and val1 < val2):
                data_copy[j], data_copy[j + 1] = data_copy[j + 1], data_copy[j]
    
    return data_copy


def ai_suggested_sort_dicts(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    AI-Suggested implementation: Sort a list of dictionaries by a specific key
    This represents what an AI code completion tool like GitHub Copilot might suggest
    
    Args:
        data: List of dictionaries to sort
        key: The key to sort by
        reverse: If True, sort in descending order
    
    Returns:
        Sorted list of dictionaries
    
    Time Complexity: O(n log n) - uses Timsort algorithm
    Space Complexity: O(n)
    """
    # AI-suggested implementation using Python's built-in sorted() function
    # This is more Pythonic and efficient
    return sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)


def optimized_sort_dicts(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    Optimized implementation with error handling and type hints
    
    Args:
        data: List of dictionaries to sort
        key: The key to sort by
        reverse: If True, sort in descending order
    
    Returns:
        Sorted list of dictionaries
    
    Raises:
        ValueError: If data is empty or key is not found in any dictionary
    """
    if not data:
        raise ValueError("Input data cannot be empty")
    
    # Check if key exists in at least one dictionary
    if not any(key in d for d in data):
        raise ValueError(f"Key '{key}' not found in any dictionary")
    
    # Use sorted with proper error handling for missing keys
    try:
        return sorted(data, key=lambda x: x.get(key, float('-inf') if not reverse else float('inf')), reverse=reverse)
    except TypeError as e:
        raise TypeError(f"Cannot compare values for key '{key}': {e}")


def generate_test_data(size: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate test data for performance comparison
    
    Args:
        size: Number of dictionaries to generate
    
    Returns:
        List of test dictionaries
    """
    test_data = []
    for i in range(size):
        test_data.append({
            'id': i,
            'score': random.randint(1, 100),
            'name': f"item_{i}",
            'priority': random.choice(['high', 'medium', 'low']),
            'timestamp': random.randint(1000000000, 2000000000)
        })
    return test_data


def performance_comparison():
    """
    Compare performance of different sorting implementations
    """
    print("=== Performance Comparison ===\n")
    
    # Generate test data of different sizes
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        print(f"Testing with {size} dictionaries:")
        test_data = generate_test_data(size)
        
        # Test manual implementation (only for smaller datasets due to O(n²) complexity)
        if size <= 500:  # Skip large datasets for bubble sort
            start_time = time.time()
            manual_result = manual_implementation_sort_dicts(test_data, 'score')
            manual_time = time.time() - start_time
            print(f"  Manual Implementation: {manual_time:.4f} seconds")
        else:
            print(f"  Manual Implementation: Skipped (too slow for {size} items)")
        
        # Test AI-suggested implementation
        start_time = time.time()
        ai_result = ai_suggested_sort_dicts(test_data, 'score')
        ai_time = time.time() - start_time
        print(f"  AI-Suggested Implementation: {ai_time:.4f} seconds")
        
        # Test optimized implementation
        start_time = time.time()
        optimized_result = optimized_sort_dicts(test_data, 'score')
        optimized_time = time.time() - start_time
        print(f"  Optimized Implementation: {optimized_time:.4f} seconds")
        
        print()


def demonstrate_functionality():
    """
    Demonstrate the functionality of all implementations
    """
    print("=== Functionality Demonstration ===\n")
    
    # Sample data
    sample_data = [
        {'name': 'Michael', 'score': 85, 'age': 25},
        {'name': 'Sylvester', 'score': 92, 'age': 30},
        {'name': 'Wambua', 'score': 78, 'age': 22},
        {'name': 'Vivian', 'score': 95, 'age': 28}
    ]
    
    print("Original data:")
    for item in sample_data:
        print(f"  {item}")
    
    print("\nSorted by score (ascending):")
    sorted_by_score = ai_suggested_sort_dicts(sample_data, 'score')
    for item in sorted_by_score:
        print(f"  {item}")
    
    print("\nSorted by age (descending):")
    sorted_by_age = ai_suggested_sort_dicts(sample_data, 'age', reverse=True)
    for item in sorted_by_age:
        print(f"  {item}")
    
    print("\nSorted by name (alphabetical):")
    sorted_by_name = ai_suggested_sort_dicts(sample_data, 'name')
    for item in sorted_by_name:
        print(f"  {item}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_functionality()
    print("\n" + "="*50 + "\n")
    performance_comparison()
    
    # Analysis Summary
    print("=== ANALYSIS SUMMARY ===")
    print("""
    EFFICIENCY COMPARISON:
    
    1. Manual Implementation (Bubble Sort):
       - Time Complexity: O(n²)
       - Space Complexity: O(n)
       - Performance: Poor for large datasets
       - Readability: Low due to nested loops
    
    2. AI-Suggested Implementation (sorted() with lambda):
       - Time Complexity: O(n log n) - uses Timsort
       - Space Complexity: O(n)
       - Performance: Excellent, leverages Python's optimized sorting
       - Readability: High, Pythonic and concise
    
    3. Optimized Implementation:
       - Same complexity as AI-suggested
       - Adds error handling and type safety
       - Better for production environments
    
    CONCLUSION:
    The AI-suggested implementation is significantly more efficient than the manual 
    bubble sort approach. It leverages Python's built-in Timsort algorithm, which 
    is highly optimized and performs well on real-world data patterns. The AI 
    suggestion demonstrates best practices by using lambda functions and the sorted() 
    builtin, resulting in code that is both more efficient and more readable.
    
    For production use, the optimized version with error handling would be preferred,
    but the AI-suggested version shows the power of modern code completion tools in
    suggesting efficient, Pythonic solutions.
    """)