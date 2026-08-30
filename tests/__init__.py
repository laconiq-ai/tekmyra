"""Makes ``tests`` a REGULAR package. This is load-bearing, not cosmetic.

Without this file ``tests`` is a NAMESPACE package, and Python resolves a
regular package of the same name anywhere on ``sys.path`` ahead of a namespace
one — *regardless of path order*. So prepending the payload root does not win,
and ``sys.path.insert(0, ROOT)`` in the verification child cannot save it.

WHAT THAT DID TO A READER. ``tests`` is among the most common top-level package
names in Python, and injecting a project root onto ``sys.path`` is exactly what
an editable install's ``.pth`` does. So any reader whose interpreter has *any*
editable-installed project carrying its own ``tests/__init__.py`` had
``import tests`` resolve to that project instead of this one. Running the
command the README puts under "without any trust in us" then printed:

    RESULT: REFUSED — the published tree ships a suite that does not pass in it.

Our own tool confidently accusing our payload of being broken, when the
collision was in the reader's environment. That is worse than a crash: a crash
is read as a bug, this is read as evidence.

MEASURED, on a staged payload with a foreign ``tests`` package injected by a
``.pth``. The clean-environment cells are identical, so this costs nothing and
is invisible except in the case it repairs:

    clean env,    no __init__.py    exit 6    288 passed, 84 skipped
    CONTAMINATED, no __init__.py    exit 2    SUITE_FAILED, 2 collection errors
    CONTAMINATED, with __init__.py  exit 6    288 passed, 84 skipped
    clean env,    with __init__.py  exit 6    288 passed, 84 skipped

The collision was diagnosed by Magnus. The four pre-existing collection errors
in the *source* suite are unrelated to this file — measured identical with and
without it — and are the documented ``uv sync --extra full-dev`` requirement.
"""
