"""Import public NxTF compiled partitions into an explicit Loihi 1 candidate.

This boundary consumes JSON, so TensorFlow and NxSDK need not be installed in
SANA-FE's environment. Public compilation is preserved; placement and the
bounded compartment execution model are supplied by this backend. This is not
an NxSDK register or instruction emulator. Original encoded IR stays in the
input artifact; the decoded graph does not model its exact synaptic packing.
"""
import hashlib
import json
import numbers
import tempfile
from importlib.resources import files
from pathlib import Path

import sanafe


def _int(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
    return int(value)


def _name(value, name):
    if not isinstance(value, str) or not value or any(c in value for c in '\n\r'):
        raise ValueError(f'{name} must be a nonempty string')
    return value


def load_nxtf_loihi1_candidate():
    """Original Loihi 1 costs with a separately named paired-compartment model."""
    source = (files('sanafe.examples') / 'loihi.yaml').read_text()
    source = source.replace('name: loihi_chip', 'name: loihi1_nxtf_candidate')
    source = source.replace('model: leaky_integrate_fire', 'model: paired_compartment_if')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'loihi1_nxtf.yaml'
        path.write_text(source)
        return sanafe.load_arch(str(path))


def validate_partition_manifest(document, *, placements=None, input_biases=None):
    """Validate and decode the supported public compiler IR before constructing a chip.

    Supports no voltage leak, fully decayed synaptic current, no refractory
    interval, and 8-bit shared-sign weights. Overflow is rejected by the soma
    model. Physical placement is an explicit experimental input. Bias overrides
    are input-frame mantissas, indexed by the input layer's logical neuron ID.
    """
    if document.get('schema') != 'nxtf-partitions-v1':
        raise ValueError('Unsupported NxTF manifest schema')
    provenance = document.get('provenance', {})
    if provenance.get('sdk_mapping') is not False or provenance.get('hardware_validation') is not False:
        raise ValueError('This importer requires public partition IR with sdk_mapping=false')
    partitions = document.get('partitions', [])
    if not partitions or len(partitions) > 128:
        raise ValueError('Expected 1..128 compiled partitions for one Loihi 1 chip')
    layers = {}
    for layer in document.get('layers', []):
        key = _name(layer['id'], 'layer id')
        if key in layers:
            raise ValueError('Duplicate layer id')
        if layer.get('signed', False):
            raise ValueError('Signed input duplication is outside the supported subset')
        if layer['type'] == 'NxInputLayer' and layer.get('input_mode', 0) != 0:
            raise ValueError('Only bias-driven NxInputLayer is supported')
        layers[key] = layer
    partition_ids = [_name(p['id'], 'partition id') for p in partitions]
    if len(set(partition_ids)) != len(partition_ids):
        raise ValueError('Duplicate partition id')
    if placements is None:
        placements = {key: [i // 4, i % 4] for i, key in enumerate(partition_ids)}
    if set(placements) != set(partition_ids):
        raise ValueError('Placements must cover exactly the compiled partitions')
    normalized_places = {}
    for key, place in placements.items():
        if not isinstance(place, (list, tuple)) or len(place) != 2:
            raise ValueError('Placement must be [tile, core]')
        normalized_places[key] = [_int(place[0], 'tile', 0, 31), _int(place[1], 'core', 0, 3)]
    if len(set(map(tuple, normalized_places.values()))) != len(partitions):
        raise ValueError('Compiled partitions must occupy distinct cores')
    input_biases = {} if input_biases is None else input_biases
    for key, values in input_biases.items():
        if key not in layers or layers[key]['type'] != 'NxInputLayer':
            raise ValueError('Bias overrides apply only to known input layers')
        if not isinstance(values, (list, tuple)):
            raise ValueError('Input bias values must be a list of integer mantissas')
        for value in values:
            _int(value, 'input bias', -4095, 4095)

    allowed_cx = {'biasExp','vThMant','enableSomaTrace','threshOp','functionalState',
                  'compartmentVoltageDecay','compartmentCurrentDecay','refractoryDelay'}
    required_cx = {'compartmentVoltageDecay':0,'compartmentCurrentDecay':4096,
                   'refractoryDelay':0,'threshOp':0,'functionalState':2,'enableSomaTrace':False}
    allowed_conn = {'numWeightBits','weightExponent','useSharedSign','synapseEncoding',
                    'delay','enableDelay','enableLearning','numDelayBits','numTagBits',
                    'weightExpSR','weightMantSR'}
    nodes, lookup, pairs, parameters = [], {}, [], {}
    inputs_seen = {key: set() for key in input_biases}
    for part in partitions:
        pid, lid = part['id'], part['layer_id']
        if lid not in layers or part['type'] != layers[lid]['type']:
            raise ValueError('Partition references an unknown or inconsistent layer')
        mode = part['reset_mode']
        if mode not in ('hard','soft') or mode != layers[lid]['reset_mode']:
            raise ValueError('Unsupported or inconsistent reset mode')
        config, conn = part['compartment_parameters'], part['connection_parameters']
        if set(config) - allowed_cx or any(config.get(k) != v for k,v in required_cx.items()):
            raise ValueError('Unsupported compartment dynamics or fields')
        if set(conn) - allowed_conn or conn.get('numWeightBits') != 8 or conn.get('useSharedSign') is not True:
            raise ValueError('Only 8-bit shared-sign connections are supported')
        for key in ('delay','enableDelay','enableLearning','numDelayBits','numTagBits'):
            if conn.get(key) != 0:
                raise ValueError(f'Unsupported connection parameter {key}')
        _int(conn.get('weightExponent'), 'weightExponent', -6, 7)
        if conn.get('synapseEncoding') not in ('sparse','dense1','runlength'):
            raise ValueError('Unsupported synapse encoding; dense2 fillers are not supported')
        threshold = 64 * _int(config.get('vThMant'), 'vThMant', 1, 131071)
        parameters[pid] = conn
        compartments = sorted(part['compartments'], key=lambda c: c['id'])
        ids = [_int(c['id'], 'compartment id', 0, 1023) for c in compartments]
        if not ids or ids != list(range(len(ids))) or len(ids) % 4:
            raise ValueError('Compartment IDs must preserve contiguous padded groups of four')
        for cx in compartments:
            cid, role = cx['id'], cx['role']
            if role not in ('hard','dendrite','soma','dummy'):
                raise ValueError('Unknown compartment role')
            if (mode=='hard' and role not in ('hard','dummy')) or (mode=='soft' and role=='hard'):
                raise ValueError('Compartment role conflicts with reset mode')
            if role=='dendrite':
                if cid % 2 or cid+1 >= len(compartments) or compartments[cid+1]['role']!='soma':
                    raise ValueError('Soft reset requires adjacent dendrite/soma pairs')
                pairs.append((pid,cid))
            if role=='soma' and (cid % 2 != 1 or compartments[cid-1]['role']!='dendrite'):
                raise ValueError('Soma lacks its adjacent dendrite')
            mant = _int(cx['bias_mantissa'], 'bias mantissa', -4095, 4095)
            exp = _int(cx['bias_exponent'], 'bias exponent', 0, 7)
            global_id = cx['global_id']
            if global_id is not None or role != 'dummy':
                global_id = _int(global_id, 'global compartment id', -1, (1<<31)-1)
            logical = cx.get('logical_id')
            if logical is None and role != 'dummy':
                logical = global_id//(2 if mode=='soft' else 1)
            if role!='dummy':
                logical = _int(logical, 'logical id', 0, (1<<31)-1)
            if lid in input_biases and role in ('hard','dendrite'):
                if logical >= len(input_biases[lid]):
                    raise ValueError('Input bias array has the wrong shape')
                mant = int(input_biases[lid][logical]); inputs_seen[lid].add(logical)
            if role in ('soma','dummy') and mant != 0:
                raise ValueError('Soma and dummy compartments require zero bias')
            node = dict(partition=pid, layer=lid, cx=cid, global_id=global_id,
                        logical_id=logical, role=role, threshold=threshold, bias=mant*(1<<exp))
            lookup[(pid,cid)] = len(nodes)
            nodes.append(node)
    for lid, seen in inputs_seen.items():
        if seen != set(range(len(input_biases[lid]))) or not seen:
            raise ValueError('Input bias array has the wrong shape')

    edges, feedback = [], {pair:[] for pair in pairs}
    for edge in document.get('edges', []):
        src = (edge['source_partition'], _int(edge['source_cx'], 'source cx', 0, 1023))
        dst = (edge['target_partition'], _int(edge['target_cx'], 'target cx', 0, 1023))
        if src not in lookup or dst not in lookup:
            raise ValueError('Edge references an unknown compartment')
        sn, dn = nodes[lookup[src]], nodes[lookup[dst]]
        if sn['role'] not in ('soma','hard') or dn['role'] not in ('dendrite','hard'):
            raise ValueError('Edge conflicts with compartment roles')
        if _int(edge['delay'], 'delay', 0, 7) != 0:
            raise ValueError('Explicit axonal/synaptic delays are unsupported')
        if _int(edge['num_weight_bits'], 'weight bits', 1, 8) != 8:
            raise ValueError('Only exact 8-bit weights are supported')
        mant = _int(edge['weight_mantissa'], 'weight mantissa', -255, 255)
        exponent = _int(edge['weight_exponent'], 'weight exponent', -6, 7)
        mode = _int(edge['sign_mode'], 'sign mode', 1, 3)
        if mode == 1 or (mode==2 and mant<0) or (mode==3 and mant>0):
            raise ValueError('Weights must use consistent shared signs')
        reset = edge['soft_reset']
        if not isinstance(reset, bool):
            raise ValueError('soft_reset must be boolean')
        target_conn = parameters[dst[0]]
        expected_exp = target_conn.get('weightExpSR' if reset else 'weightExponent')
        if exponent != expected_exp:
            raise ValueError('Edge exponent disagrees with target partition')
        if reset and mant != -target_conn.get('weightMantSR', 0):
            raise ValueError('Reset weight disagrees with target partition')
        weight = mant * (1 << (6+exponent))
        if reset:
            if dst not in feedback or src != (dst[0],dst[1]+1) or weight>=0:
                raise ValueError('Invalid recurrent reset feedback edge')
            feedback[dst].append(-weight)
        elif src == (dst[0],dst[1]+1) and dst in feedback:
            raise ValueError('Unmarked reset feedback edge')
        edges.append(dict(source=lookup[src], target=lookup[dst], weight=weight, soft_reset=reset))
    differences=[]
    for pair, values in feedback.items():
        if len(values)!=1:
            raise ValueError('Every soft-reset pair requires exactly one explicit feedback edge')
        threshold=nodes[lookup[pair]]['threshold']
        if threshold!=values[0]:
            differences.append(dict(partition=pair[0],cx=pair[1],threshold=threshold,
                                    feedback=values[0],difference=threshold-values[0]))
    digest=hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(schema='sanafe-nxtf-loihi1-execution-v1', compiled_manifest_sha256=digest,
                compiler_provenance=provenance, sdk_mapping=False, hardware_validated=False,
                execution_model='source-derived bounded paired-compartment IF candidate',
                placement_provenance='explicit SANA-FE placement', placements=normalized_places,
                input_biases={k:list(v) for k,v in input_biases.items()}, nodes=nodes, edges=edges,
                soft_reset_pairs=len(pairs), reset_differences=differences,
                costs='inherited SANA-FE Loihi 1 coefficients; pair implementation uncalibrated',
                connection_delay_updates=1, packed_synaptic_memory_execution=False,
                input_transfer_cost_included=False)


def map_compiled_partitions(document, *, placements=None, input_biases=None):
    """Return Network, Architecture, execution manifest for supported compiled IR."""
    report=validate_partition_manifest(document,placements=placements,input_biases=input_biases)
    arch=load_nxtf_loihi1_candidate()
    net=sanafe.Network()
    mapped=[]
    for partition in document['partitions']:
        pid = partition['id']
        place = report['placements'][pid]
        node_specs=[n for n in report['nodes'] if n['partition']==pid]
        group=net.create_neuron_group(pid, len(node_specs),
            soma_hw_name='loihi_lif', default_synapse_hw_name='loihi_dense_synapse',
            default_dendrite_hw_name='loihi_dendrites', log_spikes=True, log_potential=True)
        for neuron,spec in zip(group,node_specs):
            neuron.set_attributes(model_attributes=dict(pair_role=spec['role'],
                threshold=spec['threshold'],bias=spec['bias']))
            neuron.map_to_core(arch.tiles[place[0]].cores[place[1]])
            mapped.append(neuron)
    for edge in report['edges']:
        mapped[edge['source']].connect_to_neuron(mapped[edge['target']], {'weight':edge['weight']})
    report['architecture_sha256']=hashlib.sha256(json.dumps(arch.configuration(),sort_keys=True).encode()).hexdigest()
    return net,arch,report
