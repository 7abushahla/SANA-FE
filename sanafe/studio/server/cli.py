"""Command line: python -m sanafe.studio [options]."""
import argparse
from pathlib import Path

from .worker import WorkloadRef

DEFAULT_WORKLOADS = {
    'sanafe-files': WorkloadRef('sanafe.studio.engine.workload:SanafeFiles'),
}


def build_registry(workload_args, paths, extra=None):
    """Default workloads, plus NAME=module:Class entries, plus ``extra``."""
    registry = dict(DEFAULT_WORKLOADS)
    sys_path = tuple(str(Path(path).resolve()) for path in paths)
    for item in workload_args:
        name, separator, target = item.partition('=')
        if not name or not separator or ':' not in target:
            raise ValueError(f'--workload expects NAME=module:Class, got {item!r}')
        registry[name] = WorkloadRef(target, sys_path)
    registry.update(extra or {})
    return registry


def main(argv=None, extra_workloads=None):
    parser = argparse.ArgumentParser(
        prog='python -m sanafe.studio',
        description='Serve SANA-FE Studio on this machine only (127.0.0.1).')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--store-dir', type=Path, default=None,
                        help='Save each run as a trace store under this directory')
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
    uvicorn.run(create_app(registry, store_dir=args.store_dir),
                host='127.0.0.1', port=args.port, log_level='warning')
