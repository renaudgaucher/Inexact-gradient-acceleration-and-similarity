import os
import pytest
import glob
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def find_test_files():
    paths = [project_root, os.path.join(project_root, 'tests')]
    test_files = []
    for path in paths:
        pattern = os.path.join(path, 'unit_test_*.py')
        test_files.extend(glob.glob(pattern))
    return test_files

if __name__ == '__main__':
    test_files = find_test_files()
    if not test_files:
        raise SystemExit("No unit test files found.")
    else:
        print(f"Running tests from: {test_files}")
        raise SystemExit(pytest.main(test_files))