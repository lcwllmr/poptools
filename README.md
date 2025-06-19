# `poptools`: Toolbox for polynomial optimization research

[![ci](https://github.com/lcwllmr/poptools/actions/workflows/ci.yml/badge.svg)](https://github.com/lcwllmr/poptools/actions/workflows/ci.yml)

This Python package provides various ingredients for research in polynomial optimization.
The leading design principle is providing just the right set of tools to quickly and intuitively sketch algorithms and perform experiments.
For instance, while the package doesn't provide a fully-featured SDP solver, it does provide all routines needed to write one in a few lines (e.g. parsing SDP data from a file, PD line search, formulating Lagrangians, etc.). See the `examples/` directory and the section below for various showcases.

## Examples

Each item here corresponds to one showcase in the `examples/` directory.
They are all implementations of algorithms presented in the literature.

- **WIP** - `examples/hrvw1996`: The primal-dual interior point formulation that is used for instance in [gh:coin-or/Csdp](https://github.com/coin-or/Csdp). See [[Helmberg et al., 1996]](https://doi.org/10.1007/978-0-387-74759-0_292).

