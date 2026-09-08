# Latent space network models on the anisotropic Thurston geometries

Code, data and manuscript source for a study of latent space network models on the three
three-dimensional model geometries that are neither of constant curvature nor products:
the Heisenberg group (Nil), the solvable group (Sol) and the universal cover of the unit
tangent bundle of the hyperbolic plane (SL~(2,R)). The paper treats identification of
positions from one network, the posterior on the quotient by the isometry group, the
small-scale detectability of the geometry, and a directed model in which asymmetric
preferences circulate around triangles in proportion to enclosed area.

## Layout

    code/        geometry, inference, competitors and experiment drivers
    analysis/    scripts for the real-network studies (connectomes, sports, trade, games, benchmarks)
    paper/       LaTeX source of the manuscript (main.tex is the root file; supplement included)
    results/     JSON result files from which every table and number in the paper is generated
    figures/     figures used in the paper
    data/        network data with the source and licence of each collection (see data/README.md)

## Requirements

Python 3.10 or later with numpy, scipy, pandas and xlrd. Install with

    pip install -r requirements.txt

LaTeX (pdflatex, bibtex) to compile the manuscript.

## Reproducing the paper

All scripts assume the repository root as working directory.

Geometry and distance tables: `code/geometry.py` builds the exact Nil distance, the Sol and
SL~(2,R) tables, charts, Jacobians and gauge maps; `python code/geometry.py` regenerates the
tables (`code/tables.pkl`).

Simulation studies (rank witnesses, interval certificate, coverage, replicated simulations,
win maps, small-scale validation, directed simulation): `python code/experiments2.py <stage>` and
`python code/experiments3.py <stage>`; stages are listed at the top of each file and resume
from the JSON files in `results/`.

Real networks: the scripts in `analysis/` build each network from the files in `data/`,
screen it, and fit the models. For example

    python analysis/connectome_analysis.py rotfrac
    python analysis/connectome_analysis.py directed Cat1 Macaque1
    python analysis/games_analysis.py screen pokemon
    NSPLIT=3 python analysis/games_analysis.py directed pokemon nil sl2 h2r euc h3 s3 sol
    python analysis/controls.py pokemon
    FITS=euc_shared,ame python analysis/controls2.py pokemon
    python analysis/directed_mcmc.py pokemon nil

Tables: `code/make_numbers5.py` and the table-writing blocks in the analysis scripts regenerate
the LaTeX tables and macros in `paper/` from `results/`.

Manuscript: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Licence

Code: MIT (see LICENSE). Data: each collection keeps the licence of its source; see
`data/README.md`. The manuscript text is copyright of the authors.
