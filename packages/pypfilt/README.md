# pypfilt

The earlier experiment uses pypfilt.fit to execute the native particle filter, with local Dacca model callbacks and likelihood-increment accumulation. Parameter optimization uses external SciPy Nelder-Mead. We have not added IF2 or IFAD to pypfilt. The experiment measures the package PF combined with this external optimizer.

- [Readable source](experiment_code/)
- [Completed earlier results](../../experiments/legacy_dacca/pypfilt/)
- [Exact earlier calls](../../reports/package_calls_legacy/table_CN.txt)
- [Complete frozen sources and raw records](../../artifacts/experiments/legacy_dacca/pypfilt/pypfilt_dacca_formal_reproduction.tar.gz)

The reproduction archive retains the original source, environment, model, protocol, raw outcomes, validation and license notices. This package is not part of the new A/B/C experiments.
