# pypfilt

The earlier experiment uses pypfilt.fit to execute the native particle filter, with local Dacca model callbacks and likelihood-increment accumulation. Parameter optimization uses external SciPy Nelder–Mead. It does not establish a native IF2, IFAD or MLE implementation in pypfilt. No new public MLE interface is claimed.

- [Readable source](experiment_code/)
- [Completed earlier results](../../experiments/legacy_dacca/pypfilt/)
- [Exact earlier calls](../../reports/package_calls_legacy/table_CN.txt)
- [Complete frozen sources and raw records](../../artifacts/experiments/legacy_dacca/pypfilt/pypfilt_dacca_formal_reproduction.tar.gz)

The reproduction archive retains the original source, environment, model, protocol, raw outcomes, validation and license notices. This package is not silently included as an additional A/B/C public-interface experiment.
