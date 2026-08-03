# SPDX-License-Identifier: Apache-2.0

import tokenscalefp4


def test_package_exports_development_version() -> None:
    assert tokenscalefp4.__version__ == "0.1.0.dev0"
