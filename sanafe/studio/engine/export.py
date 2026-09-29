"""sanafe.viz plots of a saved run, rebuilt from its update records."""
import io

import pandas as pd

EXPORT_KINDS = ('raster', 'potential', 'energy', 'throughput', 'latency')


def _spikes(records):
    rows = [{'timestep': r.update, 'group': g, 'neuron_offset': int(o),
             'neuron_id': f'{g}.{o}'} for r in records for g, o in r.fired]
    return pd.DataFrame(rows, columns=['timestep', 'group', 'neuron_offset', 'neuron_id'])


def _potentials(records):
    frame = pd.DataFrame([r.potentials for r in records],
                         index=pd.Index([r.update for r in records], name='timestep'))
    return frame


def _performance(records):
    return pd.DataFrame([{'timestep': r.update, 'sim_time': r.step_time,
                          **{f'{kind}_energy': value for kind, value in r.energy.items()},
                          'fired': r.counts['fired'], 'spikes': r.counts['spikes'],
                          'hops': r.counts['hops']} for r in records])


def _messages(records):
    return pd.DataFrame([{'timestep': r.update, 'mid': m.mid,
                          'generation_delay': m.generation_delay,
                          'processing_delay': m.processing_delay,
                          'network_delay': m.network_delay,
                          'blocking_delay': m.blocking_delay}
                         for r in records for m in r.messages])


def export_plot(records, kind):
    """SVG bytes of one sanafe.viz plot. Raises KeyError or ValueError."""
    if kind not in EXPORT_KINDS:
        raise KeyError(f'unknown plot {kind!r}; choose one of {", ".join(EXPORT_KINDS)}')
    if not records:
        raise ValueError('the run has no updates to plot')
    import matplotlib
    matplotlib.use('Agg', force=True)  # headless and thread-safe, whatever MPLBACKEND says
    from sanafe import viz
    import matplotlib.pyplot as plt

    title = 'modeled by SANA-FE, not measured'
    if kind == 'raster':
        frame = _spikes(records)
        if frame.empty:
            raise ValueError('no logged neuron fired in this run')
        figure, _ = viz.plot_raster(frame, title=f'Spikes ({title})')
    elif kind == 'potential':
        frame = _potentials(records)
        if frame.empty or frame.shape[1] == 0:
            raise ValueError('no neuron in this run logs its membrane')
        figure, _ = viz.plot_potential(frame, title=f'Membranes ({title})')
    elif kind == 'energy':
        figure, _ = viz.plot_energy(_performance(records), title=f'Energy ({title})')
    elif kind == 'throughput':
        figure, _ = viz.plot_throughput(_performance(records), title=f'Activity ({title})')
    else:
        frame = _messages(records)
        if frame.empty:
            raise ValueError('the run has no messages')
        figure, _ = viz.plot_message_latency(frame, title=f'Message delays ({title})')
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format='svg')
    finally:
        plt.close(figure)
    return buffer.getvalue()
