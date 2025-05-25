"""Test runner for all unit tests in the project."""

import unittest
import sys
import os

def run_all_tests():
    """
    Discover and run all tests in the tests directory and its subdirectories.
    
    Returns:
        unittest.TestResult: The test results
    """
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # Get the tests directory path
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover all tests in the tests directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=tests_dir,
        pattern='test_*.py',
        top_level_dir=project_root
    )

    # Create a test runner with verbosity
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    print("\nRunning all tests...")
    print("=" * 80)
    result = test_runner.run(test_suite)
    print("=" * 80)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return non-zero exit code if there were failures or errors
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    sys.exit(run_all_tests()) 