# -*- coding: utf-8 -*-
# Copyright (C) 2016-2016 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import pathlib
import subprocess
import sys

import pytest

# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demos = []
demo_files = list(path.glob('**/demo_*.py'))
for f in demo_files:
    demos.append((f.parent, f.name))

print(demo_files)


@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    os.makedirs(os.path.join(path, "output"), exist_ok=True)
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    assert ret.returncode == 0
