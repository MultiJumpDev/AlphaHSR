#!/usr/bin/env python3
"""cli-hsr entry point.

Examples:
    python main.py list
    python main.py fight --team-a seele,bronya,sparkle,fu_xuan --team-b kafka,black_swan,luocha,himeko
    python main.py fight --team-a seele --team-b blaze_out_of_space --agent-a human
    python main.py tournament --contestants 4
    python main.py watch
"""

import sys

from cli_hsr.cli import main

if __name__ == "__main__":
    sys.exit(main())
