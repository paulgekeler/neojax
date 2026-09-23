## Benchmark Evaluators

To perform standardized and robust evaluation of neural operators, `neojax` provides two main benchmark evaluators:

### 1. Steady State Evaluator

Used for stationary or steady-state PDE problems where time is not a dynamical coordinate that requires sequential temporal propagation.

::: neojax.benchmark.evaluators.steady_state_eval.SteadyStateEvaluator

### 2. Time Dependent Evaluator

Used for time-dependent PDE problems where future states are predicted autoregressively from history. The evaluator uses a stateful scan rollout loop to evaluate accuracy over long time horizons.

::: neojax.benchmark.evaluators.time_dependent_eval.TimeDependentEvaluator