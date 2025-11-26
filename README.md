# DISCOVER Modules
## Description
This repository contains python modules that can be used together with [DISCOVER](https://github.com/hcmlab/discover).

## Installation
Every module is self_contained with a test script in the main function and a requirements.txt file in the respective folder.
Note that you might need update the requirements to your own needs (e.g. when using torch with cuda support).

## FAQ

**Q: I see SyntaxWarning or UserWarning messages from dependencies (ffmpegio, pyannote, etc.)**

A: These are harmless warnings from third-party dependencies, not errors in this codebase. They don't affect functionality. The warnings appear due to Python 3.12+ being stricter about syntax issues. The dependency maintainers need to fix their own compatibility. You can safely ignore these warnings or suppress them in your own code if desired.