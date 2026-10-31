Cuthbert: IF2 and discounted particle-filter gradients
Provisional code review, 31 October 2026

Status

This bundle contains a patch against Cuthbert commit
c737efe3098b2b5ecaa7d8b976e7cea246998b17, an example, tests, and local
validation records. The revised GPU check is job 65144479. Its results and
the new A/B/C optimization results are pending. This is not the final
benchmark release. Earlier optimizer results do not validate this interface.

What the code calls

1. IF2: cuthbert.smc.iterated_filtering.fit is added here. Each particle
   contains a model state and an estimation-scale parameter PyTree. At the
   beginning of a pass, the retained parameter swarm is perturbed and the
   model states are initialized. The propagation callback perturbs parameters
   and then calls the user's model simulator with a separate random key.
   The observation callback calls the user's log density using that particle's
   parameter. These callbacks are passed to the original
   particle_filter.build_filter. The original PF update and selected original
   resampler therefore resample states and parameters together.
   An added terminal-state scan invokes those original update functions; it
   does not call the full-trajectory cuthbert.filter function. After the last
   observation, the original resampler converts the weighted parameter swarm
   to an equally weighted swarm for the next pass. The full swarm persists,
   rather than being replaced by copies of its mean. Added outer iterations,
   Gaussian perturbations, geometric cooling and failure records implement IF2.

2. Alpha=1: the added dmop.log_likelihood interface builds the original PF
   with the original stop_gradient_decorator. That original function is
   unchanged. The new wrapper returns the terminal likelihood; the original
   propagation, weighting, resampling and derivative correction are reused.
   The default is now checkpoint=False. It preserves the original ADPF
   differentiation path in the recorded tests.

3. Alpha<1: the added discounted_stop_gradient_decorator uses the original
   resampler on stopped log weights, keeps the selected particles, and adds
   alpha * (selected_log_weight - stop_gradient(selected_log_weight)) to
   its returned weights. This correction is zero in the forward calculation.
   The discount changes the surrogate derivative, not the model likelihood.
   The original stop_gradient_decorator remains available without changes.

4. Optimization: dmop.fit and ifad.fit are new, general package interfaces.
   The former accepts an optimizer; the latter composes IF2, gradient updates
   and an independent PF evaluation. For the planned formal comparison, the
   experiment driver calls iterated_filtering.fit and dmop.log_likelihood
   directly, then applies external Optax scale_by_adam with recorded learning
   rates. The formal driver does not call dmop.fit or ifad.fit. The optimizer
   is separate from the differentiable PF.

The original cuthbert/filtering.py, cuthbert/smc/particle_filter.py and the
multinomial/systematic resampler files are byte-identical to the pinned
upstream source. full.patch contains all intended source, documentation,
packaging and test changes. Generated documentation images and caches are
excluded. The patch was applied to a clean upstream copy and every resulting
file was checked against this candidate. function_locations.json provides
the function names and line numbers for review.

Model scope

The fitting interfaces accept model callbacks and parameter PyTrees; the
Dacca model is kept outside the package implementation in validation/.
These interfaces cover bootstrap filters with resampling at every observation.
DMOP requires pathwise-differentiable process simulation. No general claim is
made for discrete simulation derivatives, adaptive resampling or guided PFs.

The frozen Dacca example has 600 monthly observations and 20 Gaussian-noise
Euler steps per month. The comparison updates 23 numerical coordinates:
18 process/observation coordinates plus five normalized initial-proportion
coordinates. The latter contain an additive-log-scale redundancy. This count
should not be described as 23 independently identifiable parameters or silently
replaced by the 15-parameter description in the introductory email. All names,
transformations and fixed parameters are explicit in validation/dacca_model.py.

Correctness evidence and unresolved GPU checks

The revised interface passes 32 local CPU tests. Full-length Dacca checks pass
with both systematic and multinomial resampling at J=32. Original/public ADPF
values and gradients also pass at all twelve planned start/replicate pairs at
J=32. These are implementation checks, not likelihood-optimization results.
Earlier local testing covered 1548 unique tests after dependency repairs; it
was not a single clean first pass and is not a claim about the GPU suite.

The earlier full GPU suite timed out after 1200 seconds with three failures:
an EnKF taper-gradient test and two Taylor smoother cases. They are unchanged
upstream tests. Job 65144329 compares original and candidate versions of these
tests. Until those records have been reviewed, an upstream-only explanation
remains unconfirmed. The full GPU suite has not passed.

A separate Dacca multinomial check exposed a gradient difference introduced
by rematerialization. One of 23 coordinates exceeded the unchanged tolerances
(rtol=1e-5, atol=1e-4). Adding rematerialization to the original ADPF reproduced
the exact same difference. With checkpoint=False, public and original ADPF
gradients are bitwise equal in that diagnostic. The original failed log,
all five diagnostic variants and the successful checks are retained.
Rematerialization remains an explicit option, with its numerical effect
documented. It is disabled for all A/B/C arms. GPU resource use must still pass
the J=5000 check before formal fitting can start.

Runtime evidence

The original native PF has already been measured separately on an A40:
T=600, J=5000, batch size1, seven synchronized timed calls, median0.393526222s;
compilation3.120166274s is separate. This is one PF call, not optimizer time.
The new GPU check measures public IF2 and gradient costs and estimated memory.
It does not repeat the native PF benchmark or turn a forecast into measured
fitting time. Full allocation time will be obtained from Slurm as well.

Planned optimization comparison, not yet measured

A: original start -> alpha1 Adam, 400 updates.
B: public IF2, 175 passes -> alpha1 Adam, 400 updates.
C: the exact same IF2 endpoint as B -> alpha0.97 Adam, 400 updates.

There are six fixed common starts and two repetitions, for36 endpoints.
B/C share the initial parameter, zero Adam state, learning rates, training
random stream and requested update count. A/B have different total computation.
The IF2 prefix is physically computed once and included in both B and C's
logical method cost. The scientific settings and iteration counts are fixed;
an allocation forecast cannot shorten them.

All endpoints, including failed fits, are retained and evaluated using an
independent frozen original PyPOMP PF at J=5000. Evaluation uses independent
keys, log(mean likelihood), MCSE and likelihood ESS, with36/72/144/288 repeats.
Training losses and perturbed IF2 likelihoods are not final scores. Low-precision
and density-floor results remain visible in the tables and raw records.

This comparison addresses Ed's question about adding reusable IF2 and IFAD
within Cuthbert. B/C isolates the discount after the same warm start; A/B
measures the effect of adding IF2 while acknowledging its extra work. Code
correctness, native PF runtime and optimization outcomes will be reported
separately, as requested in the meeting. No universal package ranking is implied.
