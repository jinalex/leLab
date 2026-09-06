"""Propagate the selected bridge constructor gate into module subprocesses."""

import runpy
import sys

from lelab.utils.follower_guard import install_guarded_followers


def main():
    module, separator, *arguments = sys.argv[1:]
    if separator != "--":
        raise ValueError("expected MODULE -- ARGUMENTS")
    install_guarded_followers()
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
