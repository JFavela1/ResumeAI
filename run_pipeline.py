"""Compatibility wrapper for the packaged pipeline CLI.

Install the package with `pip install -e .`, then prefer `resumeai-pipeline`.
"""

from resumeai.pipeline import main


if __name__ == "__main__":
    main()
