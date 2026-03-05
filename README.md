# 20251231_workflow

Project structure and organization for plant-twine interaction analysis.

## Directory Structure

```
20251231_workflow/
├── README.md
├── requirements.txt
├── overview.txt
│
├── src/                          # Source code modules
│   ├── __init__.py
│   │
│   ├── core/                     # Core data structures
│   │   ├── __init__.py
│   │   ├── exp_plant.py          # Plant state (experiment-level)
│   │   ├── exp_event.py          # Event state (interaction-level)
│   │   ├── plant_context.py      # PlantContext (runtime registry)
│   │   ├── sim_event.py          # Simulation state (interaction level)
│   │   ├── xtrm_plant.py         # Extreme Plant state (experiment-level)
│   │   └── xtrm_event.py         # Extreme Event state (interaction-level)
│   │
│   ├── io/                       # Data import/export utilities
│   │   ├── __init__.py
│   │   ├── paths.py              # Single source of truth for paths
│   │   ├── data_imports.py       # Excel, HDF5, pickle discovery & loading
│   │   ├── read_tracking_file.py # Raw tracking file parsing
│   │   └── cache.py              # Snapshotting (Plants / Events)
│   │
│   ├── geometry/                 # Geometric computations
│   │   ├── kinematics.py         # Pixel → physical geometry
│   │   └── free_shape_calcs.py   # Skeleton-based geometry (curvature, base length)
│   │
│   ├── physics/                  # Physics computations
│   │   ├── __init__.py
│   │   └── forces.py             # Force computations
│   │
│   ├── material/                 # Material property extraction
│   │   ├── __init__.py
│   │   ├── caliber_radius.py     # Radius extraction from caliber data
│   │   └── youngs_modulus.py     # Young's modulus extraction
│   │
│   ├── analysis/                 # Data analysis and fitting
│   │   ├── bootstrap.py          # Bootstrap statistical analysis
│   │   ├── exp_grouping_analysis.py  # Grouping & binning analysis
│   │   ├── exp_param_analysis.py    # Scalar reductions (E, R parameters)
│   │   ├── exp_sine_analysis.py     # Event force smoothing and fitting to sine
│   │   └── sim_analysis.py          # Simulation analysis
│   │
│   ├── plotting/                 # Visualization utilities
│   │   ├── __init__.py
│   │   ├── bootstrap.py          # Bootstrap visualization utilities
│   │   ├── exp_param_plots.py    # Parameter distribution plots
│   │   ├── exp_sine_plots.py     # Sine fit and trajectory plots
│   │   ├── grouping_plots.py     # Grouped data visualization
│   │   ├── plot_layout.py        # Plot layout and styling utilities
│   │   ├── primitives.py         # Basic plotting primitives
│   │   └── sim_plots.py          # Simulation result visualization
│   │
│   └── utils/                    # General utilities
│       ├── array_tools.py        # Array manipulation utilities
│       ├── curve_tools.py        # Curve fitting and analysis
│       ├── img_tools.py          # Image processing utilities
│       └── math_tools.py         # Mathematical function utilities
│
├── scripts/                      # Standalone analysis scripts
│   ├── bootstrap.py              # Bootstrap analysis runner
│   ├── build_exp_events.py       # Build experimental events
│   ├── build_exp_plants.py       # Build experimental plants
│   ├── build_exp_sine_fits.py    # Fit sine to experimental data
│   ├── build_sim_analysis.py     # Build simulation analysis
│   ├── build_xtrm_events.py      # Build extreme value events
│   ├── explore_exp_Egroups.py    # Explore E grouping data
│   ├── gen_Fig1.py               # Generate Figure 1
│   ├── gen_Fig2.py               # Generate Figure 2
│   ├── generate_overview.py      # Generate project overview
│   └── project_overview.py       # Project overview utility
│
├── data/                         # Data files (partially git-ignored)
│   ├── cache/                    # Cached snapshots (git-ignored)
│   ├── extreme_msup/             # Extreme support mass test data
│   ├── images/                   # Reference images and schematics
│   ├── motor_exp/                # Motor experiment data
│   └── simulations/              # Simulation data
|
└── figures/                      # Generated figures
```

## Key Modules

### Core Data Structures (`src/core/`)
- **exp_plant.py**: Represents plant state at experiment level
- **exp_event.py**: Represents individual plant-twine interaction events
- **xtrm_plant.py / xtrm_event.py**: Extreme value variants
- **sim_event.py**: Simulation-level events
- **plant_context.py**: Registry for managing plant instances at runtime

### Analysis Modules (`src/analysis/`)
- **exp_param_analysis.py**: Extract and fit scalar parameters (E, R values)
- **exp_sine_analysis.py**: Smooth forces and fit to sine curves
- **exp_grouping_analysis.py**: Grouping and binning operations
- **sim_analysis.py**: Simulation-specific analysis
- **bootstrap.py**: Bootstrap resampling for statistical analysis

### Plotting Modules (`src/plotting/`)
- **exp_param_plots.py**: Visualize parameter distributions
- **exp_sine_plots.py**: Plot sine fits and normalized trajectories
- **grouping_plots.py**: Visualize grouped data patterns
- **sim_plots.py**: Plot simulation results (forces, torque, etc.)

### Utilities (`src/utils/`)
- **array_tools.py**: Binning, length adjustment, nearest neighbor
- **curve_tools.py**: Smoothing, fitting, R², RMSE calculations
- **img_tools.py**: Image processing, skeleton point ordering
- **math_tools.py**: Curve fitting functions (sine, gaussian, power, etc.)

## Data Structure

### Experimental Data (`data/`)
- **extreme_msup/**: Extreme support mass experiments (light and stable configurations)
- **motor_exp/**: Motor-controlled experiments with tracking logs
- **simulations/**: Simulation data files
- **images/**: Reference images and schematics for documentation

### Generated Files
- **figures/**: Output folder for generated plots and figures
- **data/cache/**: Cached snapshots of Plant and Event objects (git-ignored)
