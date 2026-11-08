# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trivial import test verifying all D2 package modules import cleanly."""

import importlib


def test_import_nids_modules():
    modules = [
        "nids",
        "nids.config",
        "nids.data",
        "nids.models",
        "nids.losses",
        "nids.train",
        "nids.infer",
        "nids.metrics",
        "nids.calibrate",
        "nids.cascade",
        "nids.bench",
        "nids.viz",
    ]
    for mod in modules:
        m = importlib.import_module(mod)
        assert m is not None
