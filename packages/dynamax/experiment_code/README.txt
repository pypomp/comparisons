Dynamax full Dacca fixed-iteration comparison

Scientific scope approved: original600months /20Gaussian Euler steps /23freecoordinates /J5000. First6 frozenpaper starts,2independent repetitions each. Three groups: IF2=1950; IFAD alpha.97=175IF2+400Adam; direct original SSM.fit_sgd alpha1=425Adam without IF2. This directSGD is not paperIFAD1. No optional fourth group is submitted.

The GPU efficiency product-model batch wrapper did not pass its predefined pathwise checks, including B1 optimizer trajectories. Production does not import or call that wrapper. Each independent fit uses the original scalar DaccaSGD, unchanged SSM.fit_sgd, and the original paper Optax chain. The checkpoint optimizer calls the original Optax update directly, without a fused Adam replacement. The localPF,IF2,IFAD are extensions; Dynamax supplies the generic nativeSGD API and is not claimed to have a native Dacca PF or EM.

All full-model/forward/gradient/API checks must pass in a separate5-minute A40 gate. The12-task array depends on gate success, runs at most3GPUs concurrently, and has59m30s per task. Each task runs allthree methods on the same A40. Rotating order is fixed in advance across tasks. Upper formal allocation=5min+12*59.5min=719min, below12GPUhours. PriorLGSSM/preflights/efficiency used1028seconds, recorded separately. Actual cost projection≈8.2GPUhours; this is not measured full-run cost.

IF2 checkpoints retain full parameter swarms. SGD checkpoints retain exact original Optax moments/counts, parameters, accepted global update index and key derivation. A nonfinite fit is marked failed and its last finite point retained; other array tasks continue. Interrupted in-flight checkpoints are not automatically treated as completed work. Normal resume keeps RNG and state but can change floating-point JIT arithmetic when the minibatch count changes, so full-series bitwise reproducibility after interruption is not claimed.

Final independent scoring uses pinned original PyPOMP, J5000,R36. Initial ESS<10 or MCSE>1 triggers a predetermined extension to288, with every value retained. Low-precision and failed-fit endpoints remain visible. No endpoint is selected using training loss.

Timing: IF2 explicit compilation is reported separately. NativeSGD internal compilation is included in actualAPI fitting wall; the first true update including compile is reported, never subtracted as if it were pure compile. Independent scoring and allocation wall are separate.
