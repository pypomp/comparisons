# Meeting decisions and repository purpose

Source: the user-provided Zoom transcript, 29 October 2026, `dmop etc transcript_2026-10-29_16.02.38.txt`. The complete private transcript is not redistributed here. Times below are the transcript's time labels.

At 15:54:22–15:54:28, Ed proposed a new repository in the PyPOMP organization and used the name **comparisons**. At 15:54:46–15:55, he discussed sharing a report and a Git repository containing the code, with a possible pull request if the authors were interested. At 15:57:49–15:58:16, he considered the existing `quant` repository and preferred a separate repository containing the code, including changes that might later become a PR. Forks and submodules were discussed afterward without a settled arrangement.

The user has now authorized a local `comparisons` repository. The choice here is to preserve reviewed source snapshots and patches; it does not settle the future remote repository or submodule arrangement.

The code review needs to answer three distinct questions:

1. **Ed: what does each package actually do?** Identify the original PF, model callbacks, resampling and derivative correction being called. Describe generic parameter perturbations and IF2, and show how gradient discounting is added to the existing differentiable filter.
2. **Giles: what changed?** Provide a readable complete patch against a specified upstream version, with tests, examples and documentation. Keep the model-specific experiment outside the generic package functions.
3. **Kevin: what is supported by the evidence?** Report correctness, ordinary PF execution time and parameter-optimization outcomes separately. A matching likelihood, a smoke test or a faster PF call does not answer all three questions.

The earlier detailed Chinese notes are retained under [meeting_notes/](meeting_notes/). They are dated planning documents; subsequent authorized work and the current status take precedence over their older plans.
