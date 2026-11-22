"""Continue the finite lab catalog without changing its evidence context."""
import json
import subprocess
from pathlib import Path

from mlkit_lab import ledger, pending

REPO = 'xxammaxx/mindloop-ocr-lab'


def dispatch_next(records, run=subprocess.run):
    candidate = pending(records)
    if candidate is None:
        print('No eligible candidate (exhausted or blocked).')
        return None
    result = run(['gh', 'run', 'list', '--workflow', 'mlkit.yml', '--branch', 'main',
                  '--repo', REPO, '--limit', '100', '--json', 'status'],
                 check=True, capture_output=True, text=True)
    if any(item['status'] != 'completed' for item in json.loads(result.stdout)):
        print('An ML Kit experiment is already queued or active.')
        return None
    run(['gh', 'workflow', 'run', 'mlkit.yml', '--ref', 'main', '--repo', REPO], check=True)
    print(f'Dispatched next ML Kit experiment: {candidate}')
    return candidate


if __name__ == '__main__':
    _, state = ledger(Path('results'), Path('fixtures'))
    dispatch_next(state['records'])
