# `bad-csdp-clone`

As the title suggests, this is a clone of the classical interior-point solver CSDP (see [gh:coin-or/Csdp](https://github.com/coin-or/Csdp)).
The goal was to re-implement it in a much more readable way and drop most of the lower-level performance tweaks.

The implemented algorithm is outlined in [1] and is a modified version of the algorithm presented in [2] - a primal-dual interior point method specific to SDPs.
Main additions of the implementation over the basic idea are:

- exploitation of sparsity and block-structured matrices,
- heuristic initialization attempting to start with the right scale, and
- [Mehrotra-type predictor–corrector](https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method) steps to speed up convergence.

## Progress

As of right now, this is just a very rough (but working) implementation.
Next steps:

- extend the `VecSymDomain` class by smarter Jordan multiplication, and block-structured matrices
- go through the main solver file of CSDP and implement all the rules for detecting infeasibility etc.
- add a nice CLI to make it available to other examples
- add an integration test on [SDPLIB](https://github.com/vsdp/SDPLIB) to check correctness and compare with performance of CSDP

## References

1. Borchers, B., 1999. CSDP, A C library for semidefinite programming. *Optimization methods and Software, 11*(1-4), pp.613-623. [doi:10.1080/10556789908805765](https://doi.org/10.1080/10556789908805765).
2. Helmberg, C., Rendl, F., Vanderbei, R.J. and Wolkowicz, H., 1996. An interior-point method for semidefinite programming. *SIAM Journal on optimization, 6*(2), pp.342-361. [doi:10.1137/0806020](https://doi.org/10.1137/0806020).