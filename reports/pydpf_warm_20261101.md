# PyDPF shared IF2 completed; A/B/C submitted

The original twelve shared IF2 fits completed in job 65170880 on one A40 (gl1503). Slurm reports `COMPLETED 0:0` and 20,216 elapsed seconds. Each row completed all 175 passes, for 2,100 accepted passes and no failed rows. No fit or independent PF score was repeated.

The saved-data audit checks all twelve original starts, row seeds, full final 5,000-by-23 parameter swarms, 176-entry parameter-mean histories, 175 likelihood records, final parameter and swarm hashes, generator-state bytes and finite natural parameters. Intermediate particle swarms were not all saved. The GPU float32 swarm mean is checked against double accumulation with the existing conservative float32 summation bound; no experiment acceptance tolerance changed.

The frozen 160 source files, all 138 transferred files and 134 terminal-manifest files match their hashes. The small GPU driver checks in this job also completed under their separate deterministic test process. Production keeps the original default nondeterministic setting. The collection receipt's `scientific_audit_complete=false` records its earlier transfer stage; the subsequent separate scientific audit is complete.

The measured IF2 call time is 20,113.891332 seconds; preparation is 0.272554 seconds. Both are components of the 20,216-second allocation, not additional charges. The full final particle clouds and parameter histories remain available for review without running IF2 again.

After this audit, each previously unrun gradient phase was submitted once, using the same frozen source and physical warm results:

| Method | Job | Initialization and updates | Allocation limit |
|---|---|---|---:|
| A | 65177901 | Original starts, alpha 1, 400 Adam updates | 27,900 s |
| B | 65177902 | Audited IF2 endpoints, alpha 1, 400 Adam updates | 27,900 s |
| C | 65177903 | Same IF2 endpoints, alpha 0.97, 400 Adam updates | 27,900 s |

Each job uses one A40, four CPUs and 48 GiB host memory. B/C use identical initial parameters, empty Adam states, learning rates and paired training streams. The batch has twelve fixed slots; its elapsed time is counted once. Full rejected proposals and last valid states are retained by the existing driver. The three jobs are pending for priority in the 17:01 UTC post-submission snapshot. TFP A/B/C jobs are running in that same snapshot. Neither package has an audited gradient endpoint yet.

Including the prior 181-second startup failure, the actual warm allocation plus all three remaining allocation limits is 104,097 seconds, below the original 108,000-second formal budget. Earlier validation used 2,522 seconds in its separate budget. The failed ordinary-PF/native-DPF comparison and the old startup failure remain failures; this empirical experiment does not certify mathematical derivatives or explain the original GPU operation responsible for the differences.

Before submission, the remote source and all stopped warm files were rechecked against the audited bytes, and both the scheduler and phase directories showed no previous A/B/C attempts. Each command and receipt was saved locally and remotely. No source, model, particle count, update count or scorer changed. Independent endpoint evaluation will use the fixed original PyPOMP PF and preserve raw values, low precision, nonfinite results and density-floor flags.

The archive contains the complete stopped warm collection, frozen source, audit program and actual phase receipts. Every member hash was checked after extraction, and the extracted warm audit reproduced byte for byte without a filter call. See the [warm audit](../results/pydpf/warm_audit_20261101.json), [submission and budget record](../results/pydpf/gradient_submission_20261101.json), [recorded scheduler snapshot](../results/current_jobs.json) and [archive catalog](../archive/catalog.json).
