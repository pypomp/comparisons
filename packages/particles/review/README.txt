particles IF2 implementation review, local candidate, 2026-10-30

Read report_CN.txt for method boundaries, checks, limitations and initial failures.
Read call_chain.txt and full.patch for the actual code changes.
The archive contains a clean 103-file upstream baseline, the complete patch,
checks, copied Dacca model/input, logs and hash records. The historical driver
copy frozen_support.py is provenance only; do not execute that file as a driver.

VERIFY (Python standard library + git; does not fit or score):
  python3 verify_bundle.py

RECONSTRUCT in a fresh extracted archive directory:
  python3 verify_bundle.py --prepare
This creates upstream/ and candidate/ from the verified baseline and patch.
Do not run --prepare again in the same directory.

NUMERICAL CHECKS (optional; requires working Python dependencies):
  PYTHONPATH=candidate python3 -m pytest -q candidate/tests/test_iterated_filtering.py
  PYTHONPATH=candidate python3 candidate/book/mle/if2_stochvol.py
  PYTHONPATH=candidate python3 validation/check_original.py
  PYTHONPATH=candidate python3 validation/check_dacca.py
The last two write to validation/results unless PARTICLES_IF2_OUTPUT is set.
check_original compares baseline/candidate in separate processes.
check_dacca runs only two IF2 passes at J8 over the complete original data,
an independent recursion, and two ordinary particles PF checks. It does not
run a new formal parameter-estimation comparison or assess scoring precision.

Observed local dependencies:
Python 3.12; NumPy 1.26.4; SciPy 1.16.2; Numba 0.65.1; joblib 1.5.2;
scikit-learn 1.7.2; pytest 9.0.1. See environment.json for the full runtime record.
The local run_checks.py helper records commands and exit codes, including its
machine-specific fallback path for pygments. The direct commands above are the
portable entry points and do not require that helper or this workspace path.

No runtime ranking, formal optimization-quality result, upstream acceptance,
or complete upstream test-suite pass is claimed for the new interface.
