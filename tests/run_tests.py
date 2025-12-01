import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THETA = ROOT / 'theta.py'
TEST_DIR = ROOT / 'tests'

tests = [
    (TEST_DIR / 'factorial.th', 'fact6: 720'),
    (TEST_DIR / 'fibonacci.th', 'fib10: 55'),
    (TEST_DIR / 'ackermann.th', 'ack(2,2): 7'),
]

def run_script(path):
    proc = subprocess.run([sys.executable, str(THETA), str(path)], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    return proc.returncode, out

def main():
    failures = []
    for path, expected in tests:
        rc, out = run_script(path)
        ok = expected in out
        print(f"Running {path.name}: return={rc}, ok={ok}")
        if not ok:
            failures.append((path.name, expected, out))
    if failures:
        print('\nFailures:')
        for name, expected, out in failures:
            print(f"-- {name} expected '{expected}' but output was:\n{out}\n")
        sys.exit(1)
    print('\nAll tests passed.')

if __name__ == '__main__':
    main()
