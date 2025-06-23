# `bad-csdp-clone`

As the title suggests, this is a clone of the classical interior-point solver CSDP (see [gh:coin-or/Csdp](https://github.com/coin-or/Csdp)).
The goal was to re-implement it in a much more readable way and drop most of the lower-level performance tweaks.

The implemented algorithm is outlined in [1] and is a modified version of the algorithm presented in [2] - a primal-dual interior point method specific to SDPs.
Main additions of the implementation over the basic idea are:

- exploitation of sparsity and block-structured matrices,
- heuristic initialization attempting to start with the right scale, and
- [Mehrotra-type predictor-corrector](https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method) steps to speed up convergence.

## Progress

As of right now, this is just a very rough (but working) implementation.
Next steps:

- add support for sparse blocks - `scipy` unfortunately is not great currently, but [pydata/sparse](https://sparse.pydata.org/en/stable/) looks promising because they support higher-order sparse tensors
- go through the main solver file of CSDP and implement all the rules for detecting infeasibility etc.
- measure timings of the various operations per iteration
- add a nice CLI to make it available to other examples
- add an integration test on [SDPLIB](https://github.com/vsdp/SDPLIB) to check correctness and compare with performance of CSDP
- write explanations/derviations for each step of the solver

These things might be nice to have:

- add nice live visualization similar to tensorboard e.g. using Python's built-in HTTP server and [Chart.js](https://www.chartjs.org/); this could be super useful to integrate into the library because it would be very easy to test other algorithms as well
- use [joblib](https://joblib.readthedocs.io/en/stable/) for managing larger problems? currently we're relying only on BLAS/LAPACK built-in multi-threading, which leads to very non-optimal CPU usage in most steps

## References

1. Borchers, B., 1999. CSDP, A C library for semidefinite programming. *Optimization methods and Software, 11*(1-4), pp.613-623. [doi:10.1080/10556789908805765](https://doi.org/10.1080/10556789908805765).
2. Helmberg, C., Rendl, F., Vanderbei, R.J. and Wolkowicz, H., 1996. An interior-point method for semidefinite programming. *SIAM Journal on optimization, 6*(2), pp.342-361. [doi:10.1137/0806020](https://doi.org/10.1137/0806020).