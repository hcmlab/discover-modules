# DISCOVER Modules

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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

```bibtex
@article{hallmen2025discover,
  title     = {DISCOVER: a Data-driven Interactive System for Comprehensive
               Observation, Visualization, and ExploRation of human behavior},
  author    = {Hallmen, Tobias and Schiller, Dominik and Vehlen, Antonia and
               Eberhardt, Steffen and Baur, Tobias and Withanage, Daksitha and
               Lutz, Wolfgang and Andr{\'e}, Elisabeth},
  journal   = {Frontiers in Digital Health},
  volume    = {7},
  pages     = {1638539},
  year      = {2025},
  publisher = {Frontiers}
}
```

## License

The integration code in this repository is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0).

**Important:** Individual modules wrap or bundle third-party models and libraries that carry their own licenses. Some of these restrict commercial use. Please check the README of each module before use.

| Module | Upstream License | Commercial Use |
|---|---|---|
| blazeface | Apache 2.0 | Yes |
| blazepose | Apache 2.0 | Yes |
| crisperwhisper | CC BY-NC 4.0 | No |
| diarisation | MIT / Apache 2.0 (gated models) | Yes |
| emonet | CC BY-NC-ND 4.0 | No |
| emow2v | CC BY-NC-SA 4.0 | No (requires audEERING license) |
| facemesh | Apache 2.0 | Yes |
| gaze_clustering | MIT | Yes |
| german_sentiment | MIT | Yes |
| lens | - | - |
| libreface | USC Research License | No (research/education only) |
| opensmile | audEERING Research License | No (requires paid license) |
| rbdm | CC BY-NC-ND 4.0 | No |
| sentiment | Apache 2.0 | Yes |
| synchrony | - | - |
| voxtral | Apache 2.0 | Yes |
| w2v_bert_2 | MIT | Yes |
| whisperx | BSD-2-Clause / MIT | Yes |
| xlm_roberta | - | - |
