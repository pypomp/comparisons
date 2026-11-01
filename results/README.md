# Results

[New particles endpoint records](particles/endpoints.csv) · [Original audit](particles/audit.json) · [Recomputed summary](particles_summary.csv)

The new particles comparison has 24 endpoint records and 1,692 raw PF repeats. All 24 fits succeeded; 21 scores meet the precision requirement. Low-precision A tasks 0, 2 and 12 remain in the records and the median. There are no new GPU optimization results in this repository snapshot.

[Earlier summary](previous_summary.csv) describes 932 estimates from the completed earlier experiments. The original 968-row table, including 36 Dynamax records, and all 37,584 individual PF values are retained in the archive. Dynamax is excluded from this main summary by the user's decision. Other failures and low-precision scores remain included. Each row's finite-score count is the denominator for its median. The table is grouped by experiment, package and method, not ranked across different budgets or hardware.

Both summary CSV files can be rebuilt using `python3 scripts/reproduce.py --output build/tables`. They summarize saved endpoint estimates; this command does not run PF evaluations. Individual repetitions, checkpoints, old and corrected scoring records, original scientific figures and their plotting programs remain in [the archive](../archive/README.md).
