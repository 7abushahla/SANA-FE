"""A workload whose module sets the global start method at import, as Lava does.

Only a worker process imports this module; importing it in a test process
would change that process's start method.
"""
import multiprocessing

multiprocessing.set_start_method('fork')

from studio_helpers import ChainWorkload  # noqa: E402


class ForkingChain(ChainWorkload):
    name = 'test-forking-chain'
