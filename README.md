# Erythropoiesis Homeostasis Feedback Loop Simulation

A mathematical modeling framework for simulating red blood cell (RBC) homeostasis and erythropoiesis dynamics in health and disease states.

## Overview

This project implements a computational model of erythropoiesis (red blood cell production) based on feedback loop dynamics between:
- **Hematopoietic Stem Cells (HSCs)** - Bone marrow stem cells
- **Erythropoietin (EPO)** - Hormone regulating RBC production
- **Reticulocytes (R)** - Immature red blood cells
- **Red Blood Cells (RBCs/C)** - Mature circulating cells
- **Hemoglobin (Hbg)** - Oxygen-carrying protein

The model simulates various disease conditions affecting erythropoiesis including:
- Chronic Kidney Disease (CKD)
- Aplastic Anemia (AA)
- Hemolytic Anemia (HA)
- Iron Deficiency Anemia (IDA)
- Anemia of Chronic Disease (ACD)
- Bone Marrow Suppression (e.g., chemotherapy)

## Features

- **Dynamic simulation** of erythropoiesis using differential equations
- **Disease modeling** with customizable parameters for different pathological conditions
- **Population variability analysis** using stochastic parameter sampling
- **Nullcline analysis** for stability and equilibrium visualization
- **Experimental data comparison** with clinical datasets
- **Statistical analysis** including Wasserstein distance calculations
- **Visualization tools** for time series, phase planes, and distributions

## Project Structure

```
Homeostasis_feedback_loops/
├── Erythropoiesis_sim.py       # Core simulation model
├── disease.py                   # Disease modeling framework
├── consts.py                    # Constants and parameters
├── Erythropoiesis_sim.ipynb    # Interactive notebook
├── EPO_Hb/                      # Experimental EPO-Hemoglobin data
│   ├── AA_*.xlsx               # Aplastic anemia datasets
│   ├── ACD_*.xlsx              # Anemia of chronic disease datasets
│   ├── CKD_*.xlsx              # Chronic kidney disease datasets
│   ├── HA_*.xlsx               # Hemolytic anemia datasets
│   ├── IDA_*.xlsx              # Iron deficiency anemia datasets
│   └── norm_*.xlsx             # Normal/healthy control datasets
├── Hb_t/                        # Hemoglobin time-series data
└── *_population_simulation.csv  # Population simulation results
```

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pandas
- tqdm
- openpyxl (for Excel file handling)

### Setup

```bash
# Clone the repository
git clone https://github.com/HerutDor/Homeostasis_feedback_loops.git
cd Homeostasis_feedback_loops

# Install required packages
pip install numpy scipy matplotlib pandas tqdm openpyxl ipywidgets
```

## Usage

### Basic Simulation

```python
from disease import Disease

# Create a normal (healthy) model
normal = Disease(
    name="norm",
    modified_params={},
    hypoxia_factor=0,
    perturbation={"type": "none"}
)

# Run simulation
normal.run_simulation()

# Access results
results = normal.results
steady_state = normal.steady_state_values

# Plot time series
results.plot()
```

### Disease Modeling

```python
import consts

# Model Chronic Kidney Disease
ckd = Disease(
    name="CKD",
    modified_params={
        "e_max": consts.e_max_ckd,
        "c_normalization_epo": consts.c_normalization_ckd
    }
)
ckd.run_simulation()

# Model Aplastic Anemia
aa = Disease(
    name="AA",
    modified_params={"h_max": consts.h_max_aa}
)
aa.run_simulation()

# Model Hemolytic Anemia
ha = Disease(
    name="HA",
    modified_params={"gamma_c": consts.gamma_c_ha}
)
ha.run_simulation()
```

### Population Variability Analysis

```python
# Simulate population with parameter variability
sd_params = {
    "gamma_c": 0.2,
    "gamma_e": 0.3,
    "h_max": 0.3,
    "e_max": 0.1,
    "c_normalization_epo": 0.05,
    "a_max": 0.1,
    "d_max": 0.1,
    "k_a": 0.2,
    "k_d": 0.2
}

normal.simulate_population_variability(
    n_samples=1000,
    sd_params=sd_params,
    verbose=True
)

# Save results
normal.save_simulation_data("norm_population_simulation.csv")

# Compare with experimental data
normal.compare_to_exp_data(save_fig=True)
```

### Nullcline Analysis

```python
# Plot nullclines for stability analysis
normal.plot_nullclines()
```

### Interactive Notebook

Open `Erythropoiesis_sim.ipynb` for interactive exploration:

```bash
jupyter notebook Erythropoiesis_sim.ipynb
```

## Model Description

### Differential Equations

The model is based on the following system of ordinary differential equations:

1. **HSCs dynamics**: `dH/dt = γ_H - a(E)·H`
2. **EPO dynamics**: `dE/dt = E_max·exp(-C/C_norm) - γ_E·H·E`
3. **Reticulocytes**: `dR/dt = d(E)·H - γ_R(C)·R`
4. **RBCs**: `dC/dt = γ_R(C)·R - γ_C·C`

Where:
- `a(E)` = CFU-E proliferation rate (Michaelis-Menten)
- `d(E)` = CFU-E differentiation rate (Michaelis-Menten)
- `γ_R(C)` = Reticulocyte maturation rate (C-dependent)

### Key Parameters

Defined in `consts.py`:
- `GAMMA_C`: RBC degradation rate (1/sec)
- `GAMMA_E`: EPO degradation rate (1/(sec·cell))
- `H_MAX`: Maximum bone marrow capacity
- `E_MAX`: Maximum EPO production rate
- `A_MAX`, `K_A`: CFU-E proliferation parameters
- `D_MAX`, `K_D`: CFU-E differentiation parameters

## Experimental Data

The `EPO_Hb/` directory contains clinical datasets from various studies correlating EPO levels with hemoglobin concentrations across different disease states. Data sources are referenced in the filenames.

## Output

The simulation produces:
- Time series plots of all variables (H, E, R, C, Hbg)
- Steady-state values
- Phase plane diagrams
- Nullcline plots
- Population distribution histograms
- Statistical comparisons with experimental data

## Citation

If you use this code in your research, please cite:

```
[Add publication details when available]
```

## Related Publications

See `Supporting information - S1 file.docx` and `A minimal mathematical model of red blood cell homeostasis in anemia.docx` for detailed model description and theoretical background.

## License

[Add license information]

## Contact

For questions or collaboration:
- GitHub: [@HerutDor](https://github.com/HerutDor)

## Acknowledgments

This work models erythropoiesis homeostasis based on feedback loop principles and clinical data from multiple research groups studying various forms of anemia.
