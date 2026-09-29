"""The Studio package never imports thesis code or unexpected dependencies."""
import ast
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[2] / 'sanafe' / 'studio'
ALLOWED = {'__future__', 'argparse', 'ast', 'asyncio', 'collections', 'concurrent',
           'contextlib', 'itertools',
           'dataclasses', 'datetime', 'enum', 'importlib', 'json', 'math',
           'multiprocessing', 'os', 'pathlib', 'queue', 're', 'subprocess', 'sys',
           'tempfile', 'threading', 'typing', 'uuid', 'numpy', 'pandas', 'yaml',
           'sanafe', 'sanafecpp', 'starlette', 'uvicorn',
           'io', 'numbers', 'matplotlib'}  # matplotlib: only to close sanafe.viz figures


class TestStudioBoundaries(unittest.TestCase):
    def test_only_allowed_imports(self):
        found = {}
        for path in PACKAGE.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split('.')[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    roots = [node.module.split('.')[0]]
                else:
                    continue
                for root in roots:
                    if root not in ALLOWED:
                        found.setdefault(str(path.relative_to(PACKAGE)), []).append(root)
        self.assertEqual(found, {})


if __name__ == '__main__':
    unittest.main()
