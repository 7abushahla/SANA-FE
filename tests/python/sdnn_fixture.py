"""A small sigma-delta netx file and a frame sequence for it.

``write_sdnn_file`` writes the layer types, datasets and dtypes of lava-dl's released
``pilotnet_sdnn/network.net`` (read with h5py). Layer and neuron types are stored as in that
file, the layer type as a one-element ``S10`` array and the neuron's ``type`` and
``activation`` as variable-length UTF-8 strings. Weights and biases are float32 arrays that
hold integers, sizes and geometry are int64, ``vThMant`` is an int64 scalar and
``sigma_output`` a bool scalar.

The network, in file order, with shapes as (channels, height, width).

* 0, input (2, 8, 10), neuron SDNN with ``vThMant`` 6, no transform stored.
* 1, convolution to 3 channels, kernel 3, stride (2, 1), padding (1, 0), output (3, 4, 8).
* 2, convolution to 4 channels, kernel 3, stride 1, padding 1, output (4, 4, 8).
* 3, flatten (128, 1, 1).
* 4, dense layer of 6.
* 5, dense layer of 1 with ``sigma_output`` and no bias.

Weights are even integers in [-30, 30] and biases integers in [-20, 60], both inclusive,
drawn in layer order (weights, then bias) from ``numpy.random.default_rng(seed)``.
"""
from pathlib import Path

import h5py
import numpy as np

INPUT_SHAPE = (2, 8, 10)
#: Cores per layer with a neuron (sources, two convolutions, two dense layers).
FIXTURE_CORES = [1, 2, 1, 1, 1]


def _text(group, name, value):
    group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def _neuron(layer, *, sigma_output=False):
    group = layer.create_group("neuron")
    _text(group, "activation", "relu")
    _text(group, "type", "SDNN")
    group.create_dataset("vThMant", data=np.int64(6))
    if sigma_output:
        group.create_dataset("sigma_output", data=np.bool_(True))


def _layer(root, index, kind, shape):
    layer = root.create_group(str(index))
    layer.create_dataset("type", data=np.array([kind.encode()], dtype="S10"))
    layer.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
    return layer


def _weights(rng, shape):
    return (2 * rng.integers(-15, 15, size=shape, endpoint=True)).astype(np.float32)


def _bias(rng, n):
    return rng.integers(-20, 60, size=n, endpoint=True).astype(np.float32)


def _conv(root, index, rng, c_in, out_shape, stride, padding):
    layer = _layer(root, index, "conv", out_shape)
    c_out = out_shape[0]
    layer.create_dataset("weight", data=_weights(rng, (c_out, c_in, 3, 3)))
    layer.create_dataset("bias", data=_bias(rng, c_out))
    layer.create_dataset("dilation", data=np.array([1, 1], dtype=np.int64))
    layer.create_dataset("groups", data=np.int64(1))
    layer.create_dataset("inChannels", data=np.int64(c_in))
    layer.create_dataset("outChannels", data=np.int64(c_out))
    layer.create_dataset("kernelSize", data=np.array([3, 3], dtype=np.int64))
    layer.create_dataset("stride", data=np.asarray(stride, dtype=np.int64))
    layer.create_dataset("padding", data=np.asarray(padding, dtype=np.int64))
    _neuron(layer)


def _dense(root, index, rng, n_in, n_out, *, sigma_output=False):
    layer = _layer(root, index, "dense", (n_out,))
    layer.create_dataset("weight", data=_weights(rng, (n_out, n_in)))
    if not sigma_output:
        layer.create_dataset("bias", data=_bias(rng, n_out))
    layer.create_dataset("inFeatures", data=np.int64(n_in))
    layer.create_dataset("outFeatures", data=np.int64(n_out))
    _neuron(layer, sigma_output=sigma_output)


def write_sdnn_file(path, seed=0):
    """Write the network of the module docstring to ``path`` and return the path."""
    path = Path(path)
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        root = f.create_group("layer")
        _neuron(_layer(root, 0, "input", INPUT_SHAPE))
        _conv(root, 1, rng, 2, (3, 4, 8), (2, 1), (1, 0))
        _conv(root, 2, rng, 3, (4, 4, 8), (1, 1), (1, 1))
        _layer(root, 3, "flatten", (128, 1, 1))
        _dense(root, 4, rng, 128, 6)
        _dense(root, 5, rng, 6, 1, sigma_output=True)
    return path


def fixture_frames(steps=24, seed=0):
    """Frames ``[steps, 2, 8, 10]`` of int64.

    Row 0 is zero. Every later row is one base frame of integers in [-64, 64] plus a
    perturbation of integers in [-8, 8] drawn for that row, clipped to [-64, 64].
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(-64, 64, size=INPUT_SHAPE, endpoint=True)
    frames = np.zeros((steps, *INPUT_SHAPE), dtype=np.int64)
    for t in range(1, steps):
        frames[t] = np.clip(base + rng.integers(-8, 8, size=INPUT_SHAPE, endpoint=True),
                            -64, 64)
    return frames
