# `poptools`: Toolbox for polynomial optimization research

[![ci](https://github.com/lcwllmr/poptools/actions/workflows/ci.yml/badge.svg)](https://github.com/lcwllmr/poptools/actions/workflows/ci.yml)

This Python package provides various ingredients for research in polynomial optimization.
The leading design principle is providing just the right set of tools to quickly and intuitively sketch algorithms and perform experiments.
For instance, while the package doesn't provide a fully-featured SDP solver, it does provide all routines needed to write one in a few lines (e.g. parsing SDP data from a file, PD line search, dealing with block matrices, etc.).
See `examples/` and the section below for various showcases.
To get an overview over current set of features check out the [docs](https://lcwllmr.github.io/poptools).

## Showcases and progress

I develop the core library as needed by the particular algorithms or methods from the literature that I'm interested in.
Each item in the following list corresponds to one such showcase in the `examples/` directory.

- **[WIP]** [`bad-csdp-clone`](examples/bad-csdp-clone/): This is a simple re-implementation of [gh:coin-or/Csdp](https://github.com/coin-or/Csdp) with focus lies on code readability. The solver comes with a nice CLI and is also used in other examples.
