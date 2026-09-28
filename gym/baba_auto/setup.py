"""Build pyBaba, baba-is-auto's pybind11 module, from a utilForever/baba-is-auto checkout.

The checkout is fmri-gym's ``external/baba_auto``, at the commit the README pins;
the env reads its maps and sprites at run time. Nothing of upstream is copied
here. The path is derived from this file: pip/uv run the build in an isolated
subprocess whose cwd is not the repo root.

Upstream's own ``setup.py`` expects vcpkg to supply pybind11 and doctest, and
on Debian-family Pythons it links the static ``libpython`` it finds, which
fails. This one hands cmake the pip ``pybind11``, a stub doctest (the tests
are never built; only the ``pyBaba`` target is) and no libpython at all: an
extension module resolves Python's symbols from the interpreter that loads it.
The module lands inside the package, as ``baba_auto_gym.pyBaba``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1] / "external" / "baba_auto"
CLONE = ("git clone https://github.com/utilForever/baba-is-auto.git external/baba_auto && "
         "git -C external/baba_auto checkout <commit>  (the README pins the commit)")


class CMakeBuild(build_ext):
    """Configure and build upstream's ``pyBaba`` target into the package directory."""

    def build_extension(self, ext: Extension) -> None:
        import pybind11

        upstream = DEFAULT_REPO.resolve()
        if not (upstream / "CMakeLists.txt").is_file():
            raise RuntimeError(f"no baba-is-auto checkout at {upstream}; {CLONE}")
        out_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        build_dir = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        configure = [
            "cmake", "-S", str(upstream), "-B", str(build_dir), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_FROM_PIP=ON",
            f"-DBABA_PYTHON_EXECUTABLE={sys.executable}",
            f"-DBABA_PYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DBABA_PYTHON_EXTENSION_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX')}",
            f"-DBABA_PYTHON_OUTPUT_DIRECTORY={out_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-Ddoctest_DIR={HERE / 'cmake' / 'doctest'}",
        ]
        subprocess.check_call(configure)
        jobs = os.environ.get("NUM_JOBS", str(os.cpu_count() or 1))
        subprocess.check_call(["cmake", "--build", str(build_dir), "--target", "pyBaba",
                               "--parallel", jobs])


setup(
    ext_modules=[Extension("baba_auto_gym.pyBaba", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
