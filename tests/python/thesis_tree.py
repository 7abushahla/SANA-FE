"""Where the tests marked ``thesis`` find the Thesis repository, and what they need from it.

Five modules of this folder read files or import packages that live outside the
fork: the lava-dl checkout and the PilotNet LIF network file that
``SANA-FE-thesis/calibration/deployments/pilotnet_lif/fetch.py`` fetches, the
network facts that the same script writes into the Thesis repository, the rate
scenarios of the ``mixed-horizon-snn`` submodule, and Lava, some of them with
this project's integer IF extension (the ``Lava`` submodule). ``ROOT`` is the
folder that holds them, the environment variable ``SANAFE_THESIS_ROOT`` when it
is set and the fork's parent folder otherwise. ``conftest.py`` uses ``absent``
to fail such a module with one message, and a checkout of the fork alone runs
``pytest -m "not thesis"``, which does not collect the five modules.
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(os.environ.get("SANAFE_THESIS_ROOT") or Path(__file__).resolve().parents[3])
FETCH = "SANA-FE-thesis/calibration/deployments/pilotnet_lif/fetch.py"
LAVA_DL_SRC = ROOT / "data" / "calibration" / "lava-dl" / "src"
PILOTNET = (ROOT / "data" / "calibration" / "lava-dl" / "tutorials" / "lava" / "lib" / "dl"
            / "netx" / "pilotnet_snn" / "network.net")
FACTS_DIR = ROOT / "SANA-FE-thesis" / "calibration" / "deployments" / "pilotnet_lif"
RATES = ROOT / "mixed-horizon-snn" / "rates" / "loihi2_scenarios.json"


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


REQUIREMENTS = {
    "h5py": (lambda: _importable("h5py"),
             "the package h5py (the optional extra netx of this fork)"),
    "lava": (lambda: _importable("lava.proc.lif.process"),
             "Lava (installed from the Lava submodule of the Thesis repository)"),
    "lava_qcfs": (lambda: _importable("lava.proc.qcfs"),
                  "the integer IF extension lava.proc.qcfs of the Lava submodule"),
    "lava_dl": (LAVA_DL_SRC.is_dir, f"the lava-dl checkout {LAVA_DL_SRC} (fetched by {FETCH})"),
    "pilotnet": (PILOTNET.is_file, f"the network file {PILOTNET} (fetched by {FETCH})"),
    "facts": ((FACTS_DIR / "network_facts.json").is_file,
              f"the network facts {FACTS_DIR / 'network_facts.json'} (a file of the Thesis "
              f"repository written by {FETCH})"),
    "rates": (RATES.is_file, f"the rate scenarios {RATES} (from the mixed-horizon-snn "
                             "submodule, git submodule update --init mixed-horizon-snn)"),
}
MODULES = {
    "test_netx.py": ("h5py", "lava", "lava_dl", "pilotnet", "facts"),
    "test_lif_network_lava.py": ("h5py", "lava", "lava_dl"),
    "test_loihi2_counts.py": ("h5py", "lava_qcfs", "rates"),
    "test_graded_lava.py": ("lava", "lava_qcfs"),
    "test_sdnn_lava.py": ("lava",),
}


def absent(module):
    """The prerequisites of a module that are missing, as sentences (empty when none)."""
    return [text for key in MODULES[module] for found, text in [REQUIREMENTS[key]] if not found()]


def message(module, missing):
    absent_text = " ".join(f"{text[0].upper()}{text[1:]} is missing." for text in missing)
    return (f"{module} needs the Thesis repository around the fork (SANAFE_THESIS_ROOT, now "
            f"{ROOT}). {absent_text} Run {FETCH} and initialize the Lava and mixed-horizon-snn "
            "submodules, or run a checkout of the fork alone with pytest -m \"not thesis\".")
