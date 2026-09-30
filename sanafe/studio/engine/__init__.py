"""Scriptable Studio engine: sessions, update records, and trace storage."""
from .archdiff import architecture_diff, bundled_architectures, diff_texts
from .breakpoints import Breakpoints
from .connectivity import Connectivity
from .demos import RandomSNN
from .export import EXPORT_KINDS, export_plot
from .instrument import MAX_LOGGED_UNITS_PER_CORE, instrument_arch_yaml
from .layout import ChipLayout, CoreInfo, TileInfo, xy_path
from .runs import compare_runs, list_runs, load_run
from .records import (PROVENANCE, MappedNeuronInfo, MessageRecord, UpdateRecord,
                      build_update_record, neuron_map)
from .session import Session, SessionFault, SessionState
from .store import TRACE_FORMAT, TraceStore, from_strict_json, to_strict_json
from .workload import (BuiltWorkload, ParameterSpec, ReferenceChecker, SanafeFiles,
                       Workload, resolve_parameters)

__all__ = ['RandomSNN', 'Breakpoints', 'EXPORT_KINDS', 'export_plot', 'compare_runs', 'list_runs', 'load_run', 'architecture_diff', 'bundled_architectures', 'diff_texts', 'Connectivity', 'MAX_LOGGED_UNITS_PER_CORE', 'instrument_arch_yaml',
           'ChipLayout', 'CoreInfo', 'TileInfo', 'xy_path',
           'PROVENANCE', 'MappedNeuronInfo', 'MessageRecord', 'UpdateRecord',
           'build_update_record', 'neuron_map', 'Session', 'SessionFault',
           'SessionState', 'TRACE_FORMAT', 'TraceStore', 'from_strict_json', 'to_strict_json', 'BuiltWorkload',
           'ParameterSpec', 'ReferenceChecker', 'SanafeFiles', 'Workload',
           'resolve_parameters']
