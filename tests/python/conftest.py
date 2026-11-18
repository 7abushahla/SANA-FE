"""Collection of the test modules marked ``thesis`` (see ``thesis_tree.py``).

When the marker expression deselects ``thesis`` (``pytest -m "not thesis"``),
the three modules are not collected, so a checkout of the fork alone runs the
rest of the suite without importing what they need. Otherwise a module whose
prerequisites are missing is replaced by one test, marked ``thesis``, that
fails with a message naming every missing prerequisite. Nothing is skipped.
"""
import pytest

import thesis_tree


def _deselects_thesis(config):
    expression = config.getoption("markexpr") or ""
    if not expression:
        return False
    try:
        from _pytest.mark.expression import Expression
    except ImportError:  # the expression module of pytest moved, read the plain form
        return "not thesis" in expression
    return not Expression.compile(expression).evaluate(lambda name, **_: name == "thesis")


def pytest_ignore_collect(collection_path, config):
    if collection_path.name in thesis_tree.MODULES and _deselects_thesis(config):
        return True
    return None


class _Prerequisites(pytest.Item):
    def __init__(self, *, missing, **kwargs):
        super().__init__(**kwargs)
        self.missing = missing
        self.add_marker("thesis")

    def runtest(self):
        pytest.fail(thesis_tree.message(self.path.name, self.missing), pytrace=False)

    def reportinfo(self):
        return self.path, None, f"{self.path.name}: prerequisites of the Thesis repository"


class _ModuleWithoutPrerequisites(pytest.File):
    def __init__(self, *, missing, **kwargs):
        super().__init__(**kwargs)
        self.missing = missing

    def collect(self):
        if self.missing:
            yield _Prerequisites.from_parent(self, name="prerequisites", missing=self.missing)


def pytest_pycollect_makemodule(module_path, parent):
    if module_path.name not in thesis_tree.MODULES:
        return None
    if _deselects_thesis(parent.config):
        # A module named on the command line is collected even when ignored. It is not
        # imported, so a checkout of the fork alone never needs what it imports.
        return _ModuleWithoutPrerequisites.from_parent(parent, path=module_path, missing=[])
    missing = thesis_tree.absent(module_path.name)
    if not missing:
        return None
    return _ModuleWithoutPrerequisites.from_parent(parent, path=module_path, missing=missing)
