"""
This package provides various ingredients for the development of novel methods in polynomial optimization.
The most general object of interest is the following *POP* (polynomial optimization problem):
$$
\\begin{aligned}
\\text{minimize} \\quad & g_0(\\mathbf x) \\\\
\\text{subject to} \\quad & \\mathbf x \\in \\mathbb{R}^n \\\\
    \\quad & g_1(\\mathbf x) \\geq 0, \\dots, g_m(\\mathbf x) \\geq 0,
\\end{aligned}
$$
where $g_0, g_1, \\dots, g_m$ are real polynomials in $\\mathbb{R}[\\mathbf x]$, $\\mathbf x = (x_1, \\dots, x_n)$.
While they are generally very hard to solve due to their non-convex nature, 
POPs offer a powerful language for modeling a huge variety of problems throughout mathematics and applications.
"""

from poptools import io, linalg, opt

__all__ = ["io", "linalg", "opt"]
