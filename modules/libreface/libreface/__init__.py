"""Vendored subset of the `libreface` pip package (v0.2.0, USC Research
License, see LICENSE.rst; https://github.com/ihp-lab/LibreFace).

Only the AU-recognition and facial-expression solvers are included. The pip
package pins mediapipe==0.10.5, which has no Python >=3.12 wheels and made the
package uninstallable; this module does its own face detection/alignment and
never needed mediapipe, so the required files are shipped directly instead.
"""
