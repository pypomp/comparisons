"""Exercise only task selection; no SSH, numerical computation or collection."""
from pathlib import Path
import ast
import json
from collect_task_once import task_mapping

P = Path(__file__).resolve().parent
jobs = json.loads((P/'jobs.json').read_text())
valid = rejected = 0
for arm, job in jobs.items():
    for index in range(36):
        if index in job['indices']:
            _, selected, key = task_mapping(jobs, arm, index)
            assert selected['array_index'] == index
            assert key == f"{arm}_{selected['legacy_case']:02d}"
            valid += 1
        else:
            try:
                task_mapping(jobs, arm, index)
            except (StopIteration, AssertionError):
                rejected += 1
            else:
                raise AssertionError((arm, index))
assert (valid, rejected) == (15, 93)
ast.parse((P/'collect_task_once.py').read_text())
record = dict(passed=True, allowed_selections=valid, excluded_arm_indices=rejected,
              SSH_calls=0, model_calls=0, actual_task_collections=0,
              limitation='Only selection and syntax tested; real stopped-task transfer schema awaits first use.')
with (P/'collector_selection_controls.json').open('x') as f:
    json.dump(record, f, indent=2)
    f.write('\n')
print(json.dumps(record))
