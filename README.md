# Stacked Behind Bars Interactive Simulation (Streamlit App)

A Streamlit app for exploring how bottlenecks in Brazil's judicial process can drive pre-trial detention, prison overcrowding, and overall incarceration dynamics.

This repository contains an interactive simulation built for the broader **Stacked Behind Bars** capstone project. The app lets users test how changes in arrests, queue structure, litigation capacity, crime-group composition, sentence distributions, and comarca-level defaults affect waiting times and prison population outcomes.

## What the app does

The app models a simplified judicial processing pipeline using a discrete-event simulation. Users can:

- run single or multiple simulation trials
- pre-load parameters from Brazilian **comarcas** or use national defaults
- adjust arrests, capacities, queues, and stations
- explore crime-group probabilities and sentence-length distributions
- compare queue and station experiments
- inspect waiting-time, incarceration, and capacity-threshold charts
- view a theoretical utilisation analysis alongside simulation outputs

## Main features

- **Interactive Streamlit interface** with sidebar controls
- **Embedded simulation engine** in the app file
- **Comarca pre-load option** with editable parameters after loading
- **Scenario experimentation** for queues, service stations, and arrest pressure
- **Single-run and multi-run visualizations** with confidence intervals
- **Theoretical analysis tab** to complement simulated results

## Repository structure

A simple deployment-ready structure would look like this:

```text
.
├── stacked_behind_bars_simulation_app_v4.py
├── requirements.txt
└── README.md
```

If you want, you can later rename the main file to something shorter like `app.py`.

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run stacked_behind_bars_simulation_app_v4.py
```

If you rename the file:

```bash
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your GitHub repo.
4. Set the main file path to:
   - `stacked_behind_bars_simulation_app_v4.py`, or
   - `app.py` if you rename it.
5. Deploy.

## Dependencies

This app currently depends on:

- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `tqdm`

## Notes

- The current app embeds the simulation logic directly inside the Streamlit script.
- The comarca dataset is also embedded in the app, which makes deployment simpler because no external data file is required for the default version.
- Results are **scenario explorations**, not literal forecasts.
- Because the model is stochastic, repeated runs may produce slightly different outputs.

## Related project context

This simulation is part of the broader **Stacked Behind Bars** capstone project, which combines computational modeling and data storytelling to examine prison overcrowding in Brazil. Access the story here: https://lookerstudio.google.com/reporting/0910b53a-00f3-41c5-9061-5596e1831872

The app complements the visual story by letting users test how institutional constraints and policy levers may affect incarceration dynamics under different assumptions.

