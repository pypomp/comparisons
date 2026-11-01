# PFJAX: full Dacca check submitted

Job **65168298** was submitted on 1 November 2026 and was running on A40 node `gl1527` at the saved post-submission query. It has a 20-minute limit. [The submission record](../results/pfjax/check_v3_submission_20261101.json) includes the exact query time. No new PFJAX optimization endpoints are available.

The check compares the unchanged original `particle_filter` with the added `particle_filter_mop` at alpha 1 and 0.97 for all six starts and two validation keys per start. It uses the original Dacca model, 600 observations, 20 Euler steps per month and 5,000 particles. The existing checkpoint setting remains enabled. The original likelihood tolerance is unchanged. Raw gradients are retained; the earlier componentwise gradient rule is a diagnostic, given the [observed variation in original self-repeats](pfjax_gpu_20261101.md).

Five `iterated_filter_step` calls preserve the full parameter population. Bounded proposals using the existing external Optax Adam update measure finite values, compilation, synchronized call time and memory. These calls are validation and cost checks, not a full A/B/C fit. The ordinary PF timing benchmark is not repeated. The unchanged configured GPU test suite previously reported 72 passes.

The CPU preflight used eight particles and all twelve case identities. Its 24 saved forward comparisons agree exactly. All five IF2 calls and eight Adam proposals were finite. These are CPU results; they do not establish full-scale GPU validity. Two missing-dependency import attempts and an incorrect preflight assertion about the frozen table size were retained. The table contains 100 rows; the experiment still uses its first six rows. No numerical package source or selected start changed.

The check and the earlier 293-second diagnosis are charged to the original 11.8 GPU-hour PFJAX allowance. Reserving another twelve cases at 56 minutes gives 41,813 seconds, below the 42,480-second cap. Formal submission still requires the actual full-scale result, source and cost review. The old cancelled array has not been resubmitted.

The [setup archive](../artifacts/validation/pfjax_check_v3_setup_20261101.tar.gz) contains 445 files, including all 394 referenced source files, CPU records and unsuccessful local attempts. All archive members were checked. Its SHA256 is `5bff2963116256799aa5e4b4d1c8177523fbed4bafe6421c18222d4a82b00f87`. No GPU result is included. The 681 protected Cuthbert files were verified unchanged.
