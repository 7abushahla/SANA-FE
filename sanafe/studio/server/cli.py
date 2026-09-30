"""Command line: python -m sanafe.studio [options]."""
import argparse
from pathlib import Path

from .worker import WorkloadRef

DEFAULT_STORE = Path.home() / '.sanafe-studio' / 'runs'
DEFAULT_WORKLOADS = {
    'random-snn': WorkloadRef('sanafe.studio.engine.demos:RandomSNN'),
    'sanafe-files': WorkloadRef('sanafe.studio.engine.workload:SanafeFiles'),
}


def build_registry(workload_args, paths, extra=None):
    """``extra`` workloads, then NAME=module:Class entries, then the bundled
    defaults. The page takes its default workload, and so its default
    platform, from the first entry, so a launcher's own workloads come first;
    a later entry never overrides an earlier name."""
    registry = dict(extra or {})
    sys_path = tuple(str(Path(path).resolve()) for path in paths)
    for item in workload_args:
        name, separator, target = item.partition('=')
        if not name or not separator or ':' not in target:
            raise ValueError(f'--workload expects NAME=module:Class, got {item!r}')
        registry.setdefault(name, WorkloadRef(target, sys_path))
    for name, ref in DEFAULT_WORKLOADS.items():
        registry.setdefault(name, ref)
    return registry


def main(argv=None, extra_workloads=None):
    parser = argparse.ArgumentParser(
        prog='python -m sanafe.studio',
        description='Serve SANA-FE Studio on this machine only (127.0.0.1).')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--store-dir', type=Path, default=DEFAULT_STORE,
                        help='Save each run as a trace store under this directory '
                             '(default: %(default)s)')
    parser.add_argument('--no-store', action='store_true',
                        help='Do not save runs; Saved runs and Compare stay empty')
    parser.add_argument('--workload', action='append', default=[],
                        metavar='NAME=module:Class', help='Register a workload class')
    parser.add_argument('--path', action='append', default=[], metavar='DIR',
                        help='Import path for --workload modules')
    args = parser.parse_args(argv)
    try:
        registry = build_registry(args.workload, args.path, extra_workloads)
    except ValueError as error:
        parser.error(str(error))
    import uvicorn

    from .app import create_app

    print(f'SANA-FE Studio: http://127.0.0.1:{args.port}/', flush=True)
    store_dir = None if args.no_store else args.store_dir
    uvicorn.run(create_app(registry, store_dir=store_dir),
                host='127.0.0.1', port=args.port, log_level='warning')
