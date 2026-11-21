"""`python -m wy` 的入口：把子命令转给 wy.cli。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
