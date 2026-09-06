"""Optional operator-configured Python module wrapper, never supplied by HTTP."""

import json
import os
import sys


def python_module_command(module, arguments):
    from .follower_guard import guarded_ports

    if guarded_ports():
        arguments = [module, "--", *arguments]
        module = "lelab.scripts.guarded_module"
    raw = os.environ.get("LELAB_PYTHON_MODULE_WRAPPER")
    if raw is None:
        return [sys.executable, "-m", module, *arguments]
    prefix = json.loads(raw)
    if (
        not isinstance(prefix, list)
        or not prefix
        or any(not isinstance(item, str) or not item or "\0" in item for item in prefix)
    ):
        raise ValueError("LELAB_PYTHON_MODULE_WRAPPER must be a non-empty JSON string array")
    # Wrapper contract: <python> <prefix> MODULE -- ARGUMENTS. No shell involved.
    return [sys.executable, *prefix, module, "--", *arguments]
