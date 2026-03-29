# H11 Dependency Analysis

## Question: Where is this h11 dependency from?

## Answer: 

The `h11` dependency comes from **JupyterLab** through the following dependency chain:

```
jupyterlab → httpx → httpcore → h11
```

## Detailed Dependency Chain:

1. **JupyterLab** (`jupyterlab==4.3.4` in requirements.txt)
   - Requires: `httpx>=0.25.0,<1`
   - Purpose: JupyterLab uses httpx for HTTP communication

2. **httpx** (`httpx==0.28.1` in requirements.txt)
   - Requires: `httpcore==1.*`
   - Purpose: Modern async HTTP client library

3. **httpcore** (`httpcore==1.0.7` in requirements.txt)
   - Requires: `h11>=0.16`
   - Purpose: Low-level HTTP implementation

4. **h11** (`h11==0.14.0` in requirements.txt)
   - Purpose: Pure-Python HTTP/1.1 protocol implementation

## Why is JupyterLab needed?

This repository contains **23 Jupyter notebooks** in the `notebooks/` directory:
- 2p_preprocessing.ipynb
- FC_length_test.ipynb
- arousal_network.ipynb
- behavioral_data.ipynb
- dac.ipynb
- ephys_behavior_sync.ipynb
- fc_length_test_gc_tetrode.ipynb
- fc_length_test_simulation.ipynb
- fc_partial_correlation.ipynb
- simulate_2p_traces.ipynb
- spike_stimuli_pupil_aligned.ipynb
- t2p_cellmap.ipynb
- And others...

## Locations in Dependency Files:

- **requirements.txt**: Line 36 - `h11==0.14.0`
- **environment.yml**: Line 126 - `h11==0.14.0`

## Summary:

The `h11` dependency is **not directly used** by the 2-photon codebase. It's a transitive dependency that comes from JupyterLab, which is needed to run the numerous Jupyter notebooks in this repository for data analysis and visualization.