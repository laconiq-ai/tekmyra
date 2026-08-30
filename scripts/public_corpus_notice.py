#!/usr/bin/env python3
"""Render the open-core NOTICE from the public corpora that actually ship.

The public corpora redistribute source text. Their MIT and BSD-3-Clause
obligations therefore belong in the payload NOTICE, not in the runtime
dependency notice bundle. This generator scans every shipped public fixture and
refuses an unknown source or licence before it renders anything; adding content
cannot silently outrun its attribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_CORPUS_ROOTS = (
    "benchmarks/fixtures/public_code_v1/",
    "benchmarks/fixtures/public_diff_v1/",
    "benchmarks/fixtures/public_domain_v1/",
    "benchmarks/fixtures/public_json_v1/",
)

PUBLIC_DOMAIN_SOURCE = "congressional-record"
PUBLIC_DOMAIN_LICENSE = "public-domain-usgov"


class PublicCorpusNoticeError(RuntimeError):
    """A shipped public fixture is not covered by the NOTICE."""


@dataclass(frozen=True)
class Attribution:
    """One upstream project and the exact licence block shipped for it."""

    source: str
    display_name: str
    publisher: str
    license_id: str
    project_url: str
    license_url: str
    license_text: str


ATTRIBUTIONS = (
    Attribution(
        source="github:Textualize/rich",
        display_name="Rich",
        publisher="Textualize/rich",
        license_id="MIT",
        project_url="https://github.com/Textualize/rich",
        license_url=(
            "https://github.com/Textualize/rich/blob/"
            "9d8f9a372cc5916fd4781fec207ced7ddac2f08f/LICENSE"
        ),
        license_text="""Copyright (c) 2020 Will McGugan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
    ),
    Attribution(
        source="github:expressjs/express",
        display_name="Express",
        publisher="expressjs/express",
        license_id="MIT",
        project_url="https://github.com/expressjs/express",
        license_url=(
            "https://github.com/expressjs/express/blob/"
            "a3714473feb3d2908add734d340e7755fd85e0a3/LICENSE"
        ),
        license_text="""(The MIT License)

Copyright (c) 2009-2014 TJ Holowaychuk <tj@vision-media.ca>
Copyright (c) 2013-2014 Roman Shtylman <shtylman+expressjs@gmail.com>
Copyright (c) 2014-2015 Douglas Christopher Wilson <doug@somethingdoug.com>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.""",
    ),
    Attribution(
        source="github:nlohmann/json",
        display_name="JSON for Modern C++",
        publisher="nlohmann/json",
        license_id="MIT",
        project_url="https://github.com/nlohmann/json",
        license_url=(
            "https://github.com/nlohmann/json/blob/"
            "21af527e756435701f23e01aa8ea8dab6e050c90/LICENSE.MIT"
        ),
        license_text="""MIT License

Copyright (c) 2013-2026 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
    ),
    Attribution(
        source="github:pytest-dev/pytest",
        display_name="pytest",
        publisher="pytest-dev/pytest",
        license_id="MIT",
        project_url="https://github.com/pytest-dev/pytest",
        license_url=(
            "https://github.com/pytest-dev/pytest/blob/"
            "006884bac1cff49d2e2908c20fbbb0f0bc8e671b/LICENSE"
        ),
        license_text="""The MIT License (MIT)

Copyright (c) 2004 Holger Krekel and others

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
    ),
    Attribution(
        source="github:encode/httpx",
        display_name="HTTPX",
        publisher="encode/httpx",
        license_id="BSD-3-Clause",
        project_url="https://github.com/encode/httpx",
        license_url=(
            "https://github.com/encode/httpx/blob/"
            "b5addb64f0161ff6bfe94c124ef76f6a1fba5254/LICENSE.md"
        ),
        license_text="""Copyright © 2019, [Encode OSS Ltd](https://www.encode.io/).
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",
    ),
    Attribution(
        source="github:golang/example",
        display_name="Go example programs",
        publisher="golang/example",
        license_id="BSD-3-Clause",
        project_url="https://github.com/golang/example",
        license_url=(
            "https://github.com/golang/example/blob/"
            "7f05d217867b2af52b0a28c6d1c91df97e1b5b39/LICENSE"
        ),
        license_text="""Copyright (c) 2009 The Go Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",
    ),
)

ATTRIBUTION_BY_SOURCE = {item.source: item for item in ATTRIBUTIONS}


@dataclass(frozen=True)
class CorpusInventory:
    """Counts derived from the fixture files, never copied into the NOTICE."""

    total: int
    public_domain: int
    attribution_counts: Counter[str]

    @property
    def attribution_required(self) -> int:
        return sum(self.attribution_counts.values())


def inventory(
    project_root: Path = PROJECT_ROOT,
    roots: Iterable[str] = PUBLIC_CORPUS_ROOTS,
) -> CorpusInventory:
    """Classify every public fixture or refuse the corpus as incomplete."""
    public_domain = 0
    attribution_counts: Counter[str] = Counter()
    total = 0

    for relative_root in roots:
        root = project_root / relative_root
        if not root.is_dir():
            raise PublicCorpusNoticeError(f"public corpus root is missing: {relative_root}")
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        if not paths:
            raise PublicCorpusNoticeError(f"public corpus root is empty: {relative_root}")
        unexpected = [path for path in paths if path.suffix != ".json" or path.is_symlink()]
        if unexpected:
            names = [path.relative_to(project_root).as_posix() for path in unexpected]
            raise PublicCorpusNoticeError(
                f"public corpus contains unreviewed file type or symlink: {names}"
            )

        for path in paths:
            try:
                fixture = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicCorpusNoticeError(f"cannot read public fixture {path}: {exc}") from exc
            provenance = fixture.get("provenance")
            if not isinstance(provenance, dict):
                raise PublicCorpusNoticeError(f"public fixture has no provenance: {path}")
            source = provenance.get("source")
            license_id = provenance.get("license")
            external = provenance.get("external")
            if not isinstance(source, str) or not isinstance(license_id, str):
                raise PublicCorpusNoticeError(f"public fixture has incomplete rights labels: {path}")
            if not isinstance(external, dict) or not external.get("source_url"):
                raise PublicCorpusNoticeError(f"public fixture has no external source URL: {path}")

            total += 1
            if source == PUBLIC_DOMAIN_SOURCE:
                if license_id != PUBLIC_DOMAIN_LICENSE:
                    raise PublicCorpusNoticeError(
                        f"{path} declares {license_id!r}; expected {PUBLIC_DOMAIN_LICENSE!r}"
                    )
                public_domain += 1
                continue

            attribution = ATTRIBUTION_BY_SOURCE.get(source)
            if attribution is None:
                raise PublicCorpusNoticeError(
                    f"{path} introduces un-attributed public source {source!r}"
                )
            if license_id != attribution.license_id:
                raise PublicCorpusNoticeError(
                    f"{path} declares {license_id!r}; NOTICE binds {source} to "
                    f"{attribution.license_id!r}"
                )
            if external.get("publisher") != attribution.publisher:
                raise PublicCorpusNoticeError(
                    f"{path} publisher does not match the attribution for {source}"
                )
            attribution_counts[source] += 1

    return CorpusInventory(total, public_domain, attribution_counts)


def _attribution_block(attribution: Attribution, count: int) -> str:
    return f"""{attribution.display_name} ({attribution.publisher}) — {count} fixtures
Project: {attribution.project_url}
License: {attribution.license_id}
Upstream license at the recorded corpus revision: {attribution.license_url}

{attribution.license_text}"""


def build(project_root: Path = PROJECT_ROOT) -> str:
    """Return the complete NOTICE derived from the current public corpora."""
    observed = inventory(project_root)
    missing = [
        attribution.source
        for attribution in ATTRIBUTIONS
        if observed.attribution_counts[attribution.source] == 0
    ]
    if missing:
        raise PublicCorpusNoticeError(
            f"NOTICE attribution no longer covers any shipped fixtures: {missing}"
        )

    blocks = "\n\n".join(
        _attribution_block(item, observed.attribution_counts[item.source])
        for item in ATTRIBUTIONS
    )
    return f"""TekMyra, by LaconIQ
Copyright 2026 LaconIQ

This product includes software developed at LaconIQ.

Licensed under the Apache License, Version 2.0. See the LICENSE file for the
full text.

--------------------------------------------------------------------------------
TRADEMARKS

TekMyra™ is a trademark of LaconIQ. The Apache-2.0 licence does not grant
permission to use the TekMyra or LaconIQ names.

--------------------------------------------------------------------------------
BENCHMARK CORPORA

GENERATED CORPORA — synthetic and long_context_v1

The fixture corpora in benchmarks/fixtures/synthetic and
benchmarks/fixtures/long_context_v1 are machine-generated by the scripts named
in each fixture's own provenance block.
They contain no third-party text and no real personal data.
Identifiers that look like personal data — social-security
numbers, card numbers, email addresses at example.com — are synthetic values
written to exercise the protected-span detector.

Some fixtures carry legacy provenance labels (source: public_sample,
license: MIT, license: proprietary_laconiq_mock) that predate this repository
and describe the SHAPE the generator was imitating rather than an upstream
source. There is no upstream and no third-party attribution is owed. See the
README for the full statement.

PUBLIC CORPORA — public_code_v1, public_diff_v1, public_domain_v1 and
public_json_v1

These four corpora contain {observed.total} fixtures. Of those,
{observed.public_domain} Congressional Record fixtures are works of the United States Government and are
outside copyright under 17 U.S.C. §105; their provenance records link to the
Government Publishing Office source and policy evidence.
The remaining {observed.attribution_required} fixtures redistribute source text,
diffs or JSON from the six projects below under their stated permissive licences.
Each fixture
also carries its exact upstream source URL in its provenance block.

{blocks}

--------------------------------------------------------------------------------
THIRD-PARTY DEPENDENCIES

Third-party source is redistributed only in the public benchmark corpora
identified and attributed above. Runtime dependencies are declared in
pyproject.toml and are installed from their own distributions under their own
licences; their notices travel with those distributions.
"""


def notice_path(project_root: Path = PROJECT_ROOT) -> Path:
    """Locate NOTICE in the private staging source or in the staged payload."""
    source_path = project_root / "open_core_files" / "NOTICE"
    return source_path if source_path.parent.is_dir() else project_root / "NOTICE"


def assert_notice_current(project_root: Path = PROJECT_ROOT) -> None:
    """Refuse staging when the legal artifact differs from a fresh render."""
    path = notice_path(project_root)
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    rendered = build(project_root)
    if current != rendered:
        raise PublicCorpusNoticeError(
            f"{path.relative_to(project_root)} is stale; run "
            "scripts/public_corpus_notice.py and review the legal diff"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if NOTICE differs from a fresh corpus-derived render",
    )
    args = parser.parse_args()
    path = notice_path()
    if args.check:
        try:
            assert_notice_current()
        except PublicCorpusNoticeError as exc:
            print(exc)
            return 1
        print(f"{path.relative_to(PROJECT_ROOT)} is current")
        return 0
    path.write_text(build(), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
