"""Workloads turn validated parameters into an architecture and a mapped network."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import sanafe


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str
    default: Any = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    choices: tuple = ()
    help: str = ''

    def validate(self, value):
        if self.kind == 'int':
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{self.name}: expected an integer, got {value!r}')
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f'{self.name}: must be at least {self.minimum}, got {value}')
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f'{self.name}: must be at most {self.maximum}, got {value}')
            return value
        if self.kind == 'path':
            if not isinstance(value, (str, Path)):
                raise ValueError(f'{self.name}: expected a file path, got {value!r}')
            path = Path(value)
            if not path.is_file():
                raise ValueError(f'{self.name}: file not found: {path}')
            return path.resolve()
        if self.kind == 'choice':
            if value not in self.choices:
                raise ValueError(f'{self.name}: expected one of '
                                 f'{", ".join(map(str, self.choices))}, got {value!r}')
            return value
        raise ValueError(f'{self.name}: unknown parameter kind {self.kind!r}')


@dataclass
class BuiltWorkload:
    arch_yaml: Path
    arch: Any
    network: Any
    horizon: int
    metadata: dict = field(default_factory=dict)
    reference: Any = None
    readout: Any = None  # optional Readout: turns records into class scores (host)


class ReferenceChecker(Protocol):
    """Compares each update with reference executions of the same network.

    ``check`` returns ``{'status': 'match', 'references': [...]}``,
    ``{'status': 'unchecked', 'reason': ...}``, or ``{'status': 'mismatch',
    'reference', 'neuron', 'quantity', 'expected', 'actual', 'mismatches'}``
    naming the first differing neuron. A checker may also offer
    ``series(key)``: ``{reference: {'potential': [...], 'spike': [...]}}``
    indexed from update 1, with None where that reference has no value, and
    ``bind(session)``, called once per build with the loaded session, whose
    ``lookup`` gives the neuron order. At the aggregate trace level a record
    carries the full membranes and spikes as ``record.state`` while checked.
    """

    def check(self, record) -> dict:
        ...



class Readout(Protocol):
    """Host readout of a workload's output (for the Pipeline view).

    ``decode(record)`` runs once per update after the reference check, with
    ``record.state`` present at the aggregate level, and returns ``{'step',
    'scores', 'cumulative', 'predicted', 'reference', 'quantity'}`` (see the
    Pipeline view spec). ``bind(session)``, if present, is called once per build.
    """

    def decode(self, record): ...


class Workload(Protocol):
    name: str
    # Platform ids (sanafe.platforms) this workload can run on. Empty or absent:
    # the workload brings its own architecture and the page shows "architecture from file".
    platforms: tuple

    def parameters(self) -> tuple:
        ...

    def build(self, params: dict) -> BuiltWorkload:
        ...


def resolve_parameters(workload, supplied):
    specs = {spec.name: spec for spec in workload.parameters()}
    for name in supplied:
        if name not in specs:
            raise ValueError(f'unknown parameter {name!r} for {workload.name}')
    resolved = {}
    for name, spec in specs.items():
        if name in supplied:
            resolved[name] = spec.validate(supplied[name])
        elif spec.default is not None:
            resolved[name] = spec.validate(spec.default)
        else:
            raise ValueError(f'{name}: required')
    return resolved


class SanafeFiles:
    """Any architecture YAML and mapped network file that SANA-FE can load."""

    name = 'sanafe-files'
    platforms = ()

    def parameters(self):
        return (
            ParameterSpec('arch_yaml', 'path', help='SANA-FE architecture YAML'),
            ParameterSpec('net_file', 'path',
                          help='Mapped network: SANA-FE YAML, or a .net netlist'),
            ParameterSpec('horizon', 'int', default=10, minimum=1,
                          help='Updates executed by Run to horizon'),
        )

    def build(self, params):
        arch_yaml, net_file = Path(params['arch_yaml']), Path(params['net_file'])
        arch = sanafe.load_arch(str(arch_yaml))
        network = sanafe.load_net(str(net_file), arch,
                                  use_netlist_format=net_file.suffix == '.net')
        return BuiltWorkload(arch_yaml=arch_yaml, arch=arch, network=network,
                             horizon=params['horizon'],
                             metadata={'architecture': arch_yaml.name,
                                       'network': net_file.name})
