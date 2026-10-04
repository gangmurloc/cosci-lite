"""Minimal test runner for environments without pytest (provides approx/raises/skipif/skip/tmp_path/monkeypatch).

Usage:  python tests/run_without_pytest.py
With pytest installed, just run `pytest`.
"""
from __future__ import annotations

import importlib
import inspect
import math
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


class _Approx:
    def __init__(self, v, rel=1e-6, abs_=1e-9):
        self.v, self.rel, self.abs = v, rel, abs_

    def __eq__(self, other):
        return math.isclose(other, self.v, rel_tol=self.rel, abs_tol=self.abs)


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"did not raise {self.exc}")
        return issubclass(et, self.exc)


class _Skip(Exception):
    pass


def _skipif(cond, reason=""):
    def deco(fn):
        if cond:
            def skipped(*a, **k):
                raise _Skip(reason)
            skipped.__name__ = fn.__name__
            return skipped
        return fn
    return deco


def _skip(reason=""):
    raise _Skip(reason)


_UNSET = object()


class _MonkeyPatch:
    """setattr(obj, name, value) or setattr("pkg.mod.attr", value); undone after the test."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=_UNSET):
        if value is _UNSET:
            parts, value = target.split("."), name
            for i in range(len(parts) - 1, 0, -1):
                try:
                    target = importlib.import_module(".".join(parts[:i]))
                except ImportError:
                    continue
                for attr in parts[i:-1]:
                    target = getattr(target, attr)
                break
            name = parts[-1]
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, value in reversed(self._undo):
            setattr(target, name, value)


fake = types.ModuleType("pytest")
fake.approx = lambda v, rel=1e-6, abs=1e-9: _Approx(v, rel, abs)
fake.raises = _Raises
fake.skip = _skip
fake.mark = types.SimpleNamespace(skipif=_skipif)
sys.modules["pytest"] = fake


def main() -> int:
    failed = passed = skipped = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            kwargs = {}
            tmp = None
            patch = None
            params = inspect.signature(fn).parameters
            if "tmp_path" in params:
                tmp = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmp.name)
            if "monkeypatch" in params:
                patch = kwargs["monkeypatch"] = _MonkeyPatch()
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS {path.stem}::{name}")
            except _Skip as e:
                skipped += 1
                print(f"SKIP {path.stem}::{name} ({e})")
            except Exception:
                failed += 1
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
            finally:
                if patch:
                    patch.undo()
                if tmp:
                    tmp.cleanup()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
