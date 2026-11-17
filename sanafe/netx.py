"""Reader for the network files of lava-dl's netx (HDF5).

The reader returns every field with the meaning that lava-dl's own netx gives
it when it builds the Lava network. It depends on h5py and numpy only.
References are to lava-dl at commit b9e3691, files ``src/lava/lib/dl/netx/hdf5.py``
(written ``hdf5.py``) and ``src/lava/lib/dl/netx/utils.py`` (written ``utils.py``),
and to Lava's processes (``proc/dense/models.py``, ``proc/lif/models.py``,
``utils/weightutils.py``).

Conventions of the returned objects
-----------------------------------
* ``NetxLayer.shape`` is in the order of the file. It is (channels, height,
  width) for input and convolutional layers and (n,) for dense layers. netx
  reverses it to Lava's (x, y, c) order (``hdf5.py`` lines 255, 407).
* ``stride``, ``padding`` and ``dilation`` are (height, width) as stored. A
  scalar in the file applies to both axes (``hdf5.py`` lines 403 to 406).
* Convolution weights are (out channels, in channels, kernel height, kernel
  width) with the meaning of a cross-correlation, the PyTorch convention::

      output[o, h, w] = sum over i, kh, kw of
          weight[o, i, kh, kw] * input[i, h*stride_h + kh - pad_h,
                                          w*stride_w + kw - pad_w]

  netx flips the kernel spatially (``hdf5.py`` lines 411 and 412) and Lava then
  applies a true convolution, so the two flips cancel and the result is this
  cross-correlation.
* Neurons of a layer are indexed in flattened (channels, height, width) order.
  netx connects a flatten layer to the next dense layer through
  ``out.transpose([2, 1, 0]).flatten()`` (``hdf5.py`` lines 553 and 554). That
  order is also the column order of the first dense layer. The ``flatten`` layer
  has no neuron and no weights. It only fixes this order.
* ``weights`` holds the integers that Lava applies. For a dense layer these are
  the result of netx's ``optimize_weight_bits`` (``utils.py`` lines 80 to 148,
  called at ``hdf5.py`` lines 351 and 352) and ``weight_exp`` and ``weight_bits``
  are the values that rule returns. For a convolution netx applies no such rule
  (``hdf5.py`` lines 411 and 412). The stored integers are the weights, with
  ``weight_exp`` 0 and ``weight_bits`` 8 (the defaults of Lava's Conv).
* ``effective_weights()`` returns the integers that multiply a spike in the
  summed synaptic input. Lava's Dense shifts the accumulated sum, not each
  weight (``proc/dense/models.py`` lines 103 to 107), by a right shift for a
  negative exponent and a left shift otherwise.
* ``NetxLayer.transform`` of the input layer is the stored ``weight`` and ``bias``
  pair of the input. netx does not apply it. The dataset code of the caller does.
* ``neuron`` holds the values Lava's process receives (``hdf5.py`` lines 126 to
  149), with ``vth`` the mantissa before Lava's own left shift by 6.
* ``resets_at`` is the layer's reset phase. The layer zeroes its state at the
  start of every one-based step t with t mod ``reset_interval`` equal to this
  value, and still integrates that step's input.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# Neuron type string of the LIF of this file (hdf5.py line 130 also accepts CUBA
# with the same parameters, which this reader does not need).
_LIF_TYPE = "LOIHI"

# Neuron datasets that netx reads (hdf5.py lines 128 to 144).
_NEURON_USED = {"type", "iDecay", "vDecay", "vThMant", "messageBits"}
# Neuron datasets that netx never reads and that describe the training-time
# (slayer) neuron. Ignored: refDelay, scaleRho, tauRho, theta (the slayer
# threshold, which equals vThMant << 6) and wgtExp.
_NEURON_METADATA = {"refDelay", "scaleRho", "tauRho", "theta", "wgtExp"}

# Layer datasets by type. Redundant size attributes (inChannels, outChannels,
# kernelSize, inFeatures, outFeatures) are not read by netx. The reader checks
# them against the weights and refuses a mismatch.
_COMMON = {"type", "shape", "neuron"}
_ALLOWED = {
    "input": _COMMON | {"weight", "bias"},
    "conv": _COMMON | {"weight", "bias", "stride", "padding", "dilation", "groups",
                       "inChannels", "outChannels", "kernelSize"},
    "dense": _COMMON | {"weight", "bias", "inFeatures", "outFeatures"},
    "flatten": {"type", "shape"},
}


@dataclass
class NetxLayer:
    """One layer of a netx file. See the module docstring for the conventions."""

    kind: str
    shape: Tuple[int, ...]
    weights: Optional[np.ndarray] = None
    weight_exp: Optional[int] = None
    weight_bits: Optional[int] = None
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[Tuple[int, int]] = None
    dilation: Optional[Tuple[int, int]] = None
    groups: Optional[int] = None
    bias: Optional[np.ndarray] = None
    neuron: Optional[Dict[str, Any]] = None
    transform: Optional[Dict[str, int]] = None
    resets_at: Optional[int] = None

    def effective_weights(self) -> np.ndarray:
        """Integers that multiply a spike in the summed synaptic input.

        Raises ValueError if a negative exponent does not divide every applied
        weight, because Lava's right shift of the sum would then lose bits.
        """
        if self.weights is None:
            raise ValueError(f"a {self.kind} layer has no weights")
        weights = np.asarray(self.weights, dtype=np.int64)
        exp = int(self.weight_exp or 0)
        if exp >= 0:
            return weights << exp
        step = 1 << (-exp)
        if np.any(weights % step != 0):
            raise ValueError(
                f"applied weights are not all a multiple of {step} (2 to the power "
                f"{-exp}), so Lava's shift of the summed input by {exp} would lose "
                "bits and the layer cannot be represented by integer synapses"
            )
        return weights >> (-exp)


@dataclass
class NetxNetwork:
    """The layers of a netx file in file order, flatten layers included."""

    layers: List[NetxLayer] = field(default_factory=list)


def _read(dataset) -> np.ndarray:
    return dataset[()]


def _text(dataset) -> str:
    # utils.py lines 59 to 66: a one-element array or a scalar of ASCII bytes.
    value = _read(dataset)
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def _integers(array, what: str, layer: int) -> np.ndarray:
    # netx converts with astype(int) (utils.py line 68), which truncates.
    array = np.asarray(array)
    if not np.all(array == np.round(array)):
        raise ValueError(
            f"layer {layer}: {what} holds values with a fractional part, so the "
            "network is not an integer network"
        )
    return array.astype(np.int64)


def _pair(value, name: str, layer: int) -> Tuple[int, int]:
    # hdf5.py lines 403 to 406: a scalar applies to both axes. The pair stays in
    # file (height, width) order here, whereas netx reverses it (line 406).
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array), int(array)
    if array.shape != (2,):
        raise ValueError(f"layer {layer}: {name} has shape {array.shape}, expected "
                         "a scalar or two entries")
    return int(array[0]), int(array[1])


def _required(lg, name: str, layer: int):
    # netx reads these with layer_config[name] (hdf5.py lines 407 to 416), which
    # raises KeyError when the dataset is absent. There is no default.
    if name not in lg:
        raise ValueError(f"layer {layer}: required attribute '{name}' is missing "
                         "(netx has no default for it)")
    return _read(lg[name])


def _check_size(lg, name: str, expected, layer: int) -> None:
    if name in lg:
        stored = tuple(int(x) for x in np.atleast_1d(_read(lg[name])))
        if stored != tuple(int(x) for x in np.atleast_1d(expected)):
            raise ValueError(f"layer {layer}: attribute '{name}' is {stored} but "
                             f"the weights imply {tuple(np.atleast_1d(expected))}")


def _optimize_weight_bits(weight: np.ndarray, layer: int):
    """netx's utils.optimize_weight_bits (lines 80 to 148), reproduced.

    Returns the applied weights, the number of weight bits and the exponent.
    """
    if not np.any(weight):
        raise ValueError(f"layer {layer}: dense weights are all zero, so netx's "
                         "weight rule has no scale")
    max_weight = np.max(weight)
    min_weight = np.min(weight)
    # utils.py lines 104 to 121
    if max_weight < 0:
        is_signed = 0
        scale = -256 / min_weight
    elif min_weight >= 0:
        is_signed = 0
        scale = 255 / max_weight
    else:
        is_signed = 1
        # A zero maximum gives an infinite positive scale, as in netx (line 115).
        with np.errstate(divide="ignore"):
            scale = min(127 / max_weight, -128 / min_weight)
    scale_bits = int(np.floor(np.log2(scale)) + is_signed)
    # utils.py lines 125 to 134: the largest power of two (at most 8) that
    # divides every weight.
    n = 8
    while n > 0 and np.sum(np.abs(weight / 2.0**n - np.round(weight / 2.0**n))) != 0:
        n -= 1
    n -= is_signed
    bits = 8 - scale_bits - n
    exponent = -scale_bits
    if bits > 8:
        # Lava's truncate_weights would shift by a negative amount (utils/
        # weightutils.py lines 236 to 239), which is not defined.
        raise ValueError(
            f"layer {layer}: netx's weight rule gives {bits} weight bits, more than "
            "the 8 that Lava's Dense supports"
        )
    applied = np.left_shift(weight.astype(np.int32), scale_bits).astype(np.int64)
    # Lava's Dense then clips to 8 bits and truncates low bits (proc/dense/models.py
    # lines 87 to 90, utils/weightutils.py lines 204 to 247 and 250 to 295). It
    # decides the sign mode itself from the weights (determine_sign_mode, line 21).
    # The reader refuses a layer on which that changes a weight.
    lava_mode = ("excitatory" if np.min(applied) >= 0
                 else "inhibitory" if np.max(applied) <= 0 else "mixed")
    signed = applied if lava_mode != "inhibitory" else -applied
    mixed = int(lava_mode == "mixed")
    clipped = np.clip(signed, -(2**8) * mixed, 2**8 - 1)
    drop = 8 - bits + mixed
    truncated = (clipped >> drop) << drop
    if not np.array_equal(truncated, signed):
        raise ValueError(
            f"layer {layer}: Lava's Dense would clip or truncate the applied weights "
            f"({bits} bits, exponent {exponent}), so they are not the integers "
            "this reader would return"
        )
    return applied, int(bits), int(exponent)


def _neuron(group, layer: int, reset_interval, reset_offset) -> Dict[str, Any]:
    kind = _text(group["type"])
    if kind != _LIF_TYPE:
        raise ValueError(
            f"layer {layer}: neuron type '{kind}' is not implemented, only the "
            f"'{_LIF_TYPE}' LIF of netx files"
        )
    unknown = set(group.keys()) - _NEURON_USED - _NEURON_METADATA
    if unknown:
        raise ValueError(f"layer {layer}: neuron attribute(s) {sorted(unknown)} "
                         "are not implemented")
    if "messageBits" in group and int(_read(group["messageBits"])) != 0:
        raise ValueError(f"layer {layer}: neuron messageBits is not zero, so "
                         "graded spikes are not implemented")
    return {
        # hdf5.py lines 141 to 144
        "du": int(_read(group["iDecay"])) - 1,
        "dv": int(_read(group["vDecay"])),
        "vth": int(_read(group["vThMant"])),
        "bias_exp": 6,
        "reset_interval": reset_interval,
        "reset_offset": reset_offset,
    }


def _resets_at(reset_interval, reset_offset) -> Optional[int]:
    # Lava stores the offset modulo the interval (proc/lif/models.py line 412)
    # and zeroes u and v when time_step mod interval equals it (lines 434 to 436).
    if reset_interval is None:
        return None
    return int(reset_offset) % int(reset_interval)


def _check_keys(group, kind: str, index: int) -> None:
    if "delay" in group:
        raise ValueError(f"layer {index}: a stored delay is not implemented")
    unknown = set(group.keys()) - _ALLOWED[kind]
    if unknown:
        raise ValueError(f"layer {index} ({kind}): attribute(s) {sorted(unknown)} "
                         "are not implemented")


def read_netx(path, reset_interval: Optional[int] = None,
              reset_offset: int = 0) -> NetxNetwork:
    """Read a netx HDF5 file.

    ``reset_interval`` and ``reset_offset`` are the arguments of netx's
    ``hdf5.Network``. They are not stored in the file. As in netx, the first
    layer with a neuron receives offset ``reset_offset + 1`` (time starts from 1,
    ``hdf5.py`` line 485) and every further layer with a neuron receives one more
    (lines 497, 522, 550). Flatten layers do not advance it.
    """
    layers: List[NetxLayer] = []
    next_offset = reset_offset + 1
    with h5py.File(path, "r") as f:
        group = f["layer"]
        for index in range(len(group)):
            lg = group[str(index)]
            kind = _text(lg["type"])
            if kind not in _ALLOWED:
                raise ValueError(f"layer {index}: layer type '{kind}' is not "
                                 "implemented")
            _check_keys(lg, kind, index)
            shape = tuple(int(x) for x in _read(lg["shape"]))
            layer = NetxLayer(kind=kind, shape=shape)

            if kind == "flatten":
                layers.append(layer)
                continue

            if "neuron" not in lg:
                if kind != "input":
                    raise ValueError(f"layer {index} ({kind}): no neuron group")
            else:
                offset = None if reset_interval is None else next_offset
                layer.neuron = _neuron(lg["neuron"], index, reset_interval, offset)
                layer.resets_at = _resets_at(reset_interval, offset)
                next_offset += 1

            if kind == "input":
                # hdf5.py lines 261 to 270. The transform is stored on the block
                # and never applied by netx.
                weight = int(_read(lg["weight"])) if "weight" in lg else 64
                bias = int(_read(lg["bias"])) if "bias" in lg else 0
                layer.transform = {"weight": weight, "bias": bias}
            elif kind == "conv":
                _read_conv(lg, layer, index)
            else:
                _read_dense(lg, layer, index)
            layers.append(layer)
    return NetxNetwork(layers=layers)


def _read_conv(lg, layer: NetxLayer, index: int) -> None:
    layer.stride = _pair(_required(lg, "stride", index), "stride", index)
    layer.padding = _pair(_required(lg, "padding", index), "padding", index)
    layer.dilation = _pair(_required(lg, "dilation", index), "dilation", index)
    if layer.dilation != (1, 1):
        raise ValueError(f"layer {index}: dilation {layer.dilation} is not "
                         "implemented, only 1")
    groups = int(np.asarray(_required(lg, "groups", index)).reshape(-1)[0])
    if groups != 1:
        raise ValueError(f"layer {index}: groups {groups} is not implemented, only 1")
    layer.groups = groups
    weight = _integers(_read(lg["weight"]), "weight", index)
    # hdf5.py line 412 reshapes to the first four axes, which allows a trailing 1.
    if weight.ndim == 5 and weight.shape[4] == 1:
        weight = weight[..., 0]
    if weight.ndim != 4:
        raise ValueError(f"layer {index}: convolution weight has shape "
                         f"{weight.shape}, expected 4 axes")
    # Lava's fixed-point Conv holds weights as int32 with precision 8
    # (proc/conv/models.py line 76), a signed 8-bit range. Lava does not clip them.
    if weight.min() < -128 or weight.max() > 127:
        raise ValueError(f"layer {index}: convolution weights span "
                         f"[{weight.min()}, {weight.max()}], outside the signed 8-bit "
                         "range of Lava's fixed-point Conv")
    _check_size(lg, "outChannels", weight.shape[0], index)
    _check_size(lg, "inChannels", weight.shape[1], index)
    _check_size(lg, "kernelSize", weight.shape[2:], index)
    layer.weights = weight
    layer.weight_exp, layer.weight_bits = 0, 8
    if "bias" in lg:
        layer.bias = _integers(_read(lg["bias"]), "bias", index).reshape(-1)


def _read_dense(lg, layer: NetxLayer, index: int) -> None:
    n_out = int(np.prod(layer.shape))
    layer.shape = (n_out,)  # hdf5.py line 312
    weight = _integers(_read(lg["weight"]), "weight", index)
    if weight.ndim == 1:
        weight = weight.reshape(n_out, -1)  # hdf5.py lines 348 and 349
    elif weight.ndim != 2:
        # netx passes other shapes on and Lava's Dense rejects them
        # (proc/dense/process.py lines 82 to 85).
        raise ValueError(f"layer {index}: dense weight has shape {weight.shape}, "
                         "expected 1 or 2 axes")
    _check_size(lg, "outFeatures", weight.shape[0], index)
    _check_size(lg, "inFeatures", weight.shape[1], index)
    layer.weights, layer.weight_bits, layer.weight_exp = _optimize_weight_bits(
        weight, index)
    if "bias" in lg:
        layer.bias = _integers(_read(lg["bias"]), "bias", index).reshape(-1)

