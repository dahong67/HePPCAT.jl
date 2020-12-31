# HePPCAT: HEteroscedastic Probabilistic PCA Technique

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/dahong67/HePPCAT.jl/workflows/CI/badge.svg)](https://github.com/dahong67/HePPCAT.jl/actions)
[![Coverage](https://codecov.io/gh/dahong67/HePPCAT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dahong67/HePPCAT.jl)

> :wave: *This package provides research code and work is ongoing.
> If you are interested in using it in your own research,
> **I'd love to hear from you and collaborate!**
> Feel free to write: dahong67@wharton.upenn.edu*

Please cite the following paper for this technique:
> todo: add reference

## What is Heteroscedastic Probabilistic PCA (HePPCAT)?

**HePPCAT** is a probabilistic **Principal Component Analysis (PCA)** technique **for data that has samples with heterogeneous quality**,
i.e., noise that is *[heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity) across samples*.
It's not just a ["cool cat"](https://en.wiktionary.org/wiki/hepcat)!

In the following illustration,
sample data points
consist of a first group of *noisier samples*
and a second group of *cleaner samples*.

![2D illustration](/demo/illustration-2D.png)

Homoscedastic PPCA estimates only one noise variance for all the data
and treats them uniformly,
which can degrade performance.

**HePPCAT estimates *separate* noise variances for each group
*jointly* with the latent components
to improve recovery.**

## How does it work?

todo: add short description
