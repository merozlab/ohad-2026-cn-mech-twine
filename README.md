# ohad-2026-cn-mech-twine

Project structure and organization for plant-twine interaction analysis for article "Embodied mechanical sensing drives support selection in twining plants" , Ohad et al 2026


## Overview

This repository contains the analysis pipeline used to process experimental and simulation data of plant stem–twine interactions. The workflow includes data import, geometric reconstruction, mechanical analysis, statistical processing, and figure generation.

All figures in the manuscript can be reproduced using the scripts in the `scripts/` directory.

## Repository Structure

src/ Core analysis modules
scripts/ Scripts used to generate analysis results and figures
data/ Experimental and simulation data
figures/ Output directory for generated figures


Key components include:

- **Geometry analysis** – extraction of plant stem geometry from image tracking data  
- **Mechanical analysis** – computation of forces and material parameters  
- **Statistical analysis** – bootstrap resampling and parameter grouping  
- **Visualization tools** – generation of manuscript figures

## Reproducing the Figures

After installing dependencies:

pip install -r requirements.txt

Run the figure generation scripts

Generated figures will appear in the `figures/` directory.

## Data

Experimental, simulation, and reference data are located in:
\data

Cached intermediate results are stored in `data/cache/` and are not tracked by Git.

## Requirements

Python dependencies are listed in:
requirements.txt

## Citation

If you use this code or data, please cite the associated article:


Amir Ohad, Amir Porat, Yasmine Meroz; 2026; Embodied mechanical sensing drives support selection in twining plants; 


