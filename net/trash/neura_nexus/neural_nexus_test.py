#!/usr/bin/env python3
"""
Test script for Neural Nexus IDE Auto-Heal functionality
This script intentionally contains errors to demonstrate the auto-healing feature
"""

# Intentional syntax error - missing colon
def calculate_fibonacci(n)
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

# Intentional runtime error - undefined variable
def process_data():
    data = [1, 2, 3, 4, 5]
    # Using undefined variable 'multiplier'
    result = [x * multiplier for x in data]
    return result

# Intentional import error - non-existent module
import non_existent_module

# Intentional type error
def divide_numbers():
    # Trying to divide string by number
    result = "10" / 2
    return result

# Intentional attribute error
def use_math_function():
    import math
    # math module doesn't have 'sgrt' (should be 'sqrt')
    result = math.sgrt(16)
    return result

# Main function with multiple errors
def main():
    print("Testing Neural Nexus IDE Auto-Heal Feature")
    print("-" * 50)
    
    # Test 1: Fibonacci
    print("\nTest 1: Fibonacci sequence")
    fib_sequence = calculate_fibonacci(10)
    print(f"First 10 Fibonacci numbers: {fib_sequence}")
    
    # Test 2: Process data
    print("\nTest 2: Process data")
    processed = process_data()
    print(f"Processed data: {processed}")
    
    # Test 3: Division
    print("\nTest 3: Division")
    division_result = divide_numbers()
    print(f"Division result: {division_result}")
    
    # Test 4: Math function
    print("\nTest 4: Math function")
    math_result = use_math_function()
    print(f"Square root of 16: {math_result}")
    
    # Test 5: Using the non-existent module
    print("\nTest 5: Using imported module")
    non_existent_module.some_function()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()

"""
Expected behavior when running in Neural Nexus IDE with Auto-Heal enabled:

1. First run: Syntax error on line 8 (missing colon)
   - Auto-heal will add the missing colon

2. Second run: Import error for 'non_existent_module'
   - Auto-heal will either remove the import or replace with a valid module

3. Third run: Runtime error in process_data() - undefined 'multiplier'
   - Auto-heal will define multiplier or modify the code

4. Fourth run: Type error in divide_numbers() - can't divide string by int
   - Auto-heal will convert string to int

5. Fifth run: Attribute error in use_math_function() - 'sgrt' should be 'sqrt'
   - Auto-heal will correct the function name

6. Final run: Script should execute successfully!

This demonstrates the power of the auto-healing feature that will:
- Fix syntax errors
- Resolve import issues
- Handle runtime errors
- Correct typos in function/attribute names
- Convert types as needed
"""
