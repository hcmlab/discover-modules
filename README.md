# DISCOVER Modules
## Description
This repository contains Python modules that can be used together with [DISCOVER](https://github.com/hcmlab/discover).
You can find full framework documentation in the main repository:
https://github.com/hcmlab/discover

## Installation
Every module is self-contained with a test script in the main function and a `requirements.txt` file in the respective folder.
You might need to adapt requirements to your environment (e.g., CUDA-enabled torch builds).

## FAQ

**Q: I see SyntaxWarning or UserWarning messages from dependencies (ffmpegio, pyannote, etc.)**

A: These are harmless warnings from third-party dependencies, not errors in this codebase. They don't affect functionality. The warnings appear due to Python 3.12+ being stricter about syntax issues. The dependency maintainers need to fix their own compatibility. You can safely ignore these warnings or suppress them in your own code if desired.

## Citation
If you use DISCOVER modules, please cite the DISCOVER paper:

```
@article{hallmen2025discover,
  title={DISCOVER: a Data-driven Interactive System for Comprehensive Observation, Visualization, and ExploRation of human behavior},
  author={Hallmen, Tobias and Schiller, Dominik and Vehlen, Antonia and Eberhardt, Steffen and Baur, Tobias and Withanage, Daksitha and Lutz, Wolfgang and Andr{\'e}, Elisabeth},
  journal={Frontiers in Digital Health},
  volume={7},
  pages={1638539},
  publisher={Frontiers}
}
```
