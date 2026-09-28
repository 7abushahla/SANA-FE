"""Manifest validation independent of optional TensorFlow/NxSDK installations."""
import copy
import importlib.util
import unittest


def example():
    params = dict(biasExp=6, vThMant=255, enableSomaTrace=False, threshOp=0,
                  functionalState=2, compartmentVoltageDecay=0,
                  compartmentCurrentDecay=4096, refractoryDelay=0)
    conn = dict(numWeightBits=8, weightExponent=0, useSharedSign=True,
                synapseEncoding='dense1', delay=0, enableDelay=0,
                enableLearning=0, numDelayBits=0, numTagBits=0,
                weightExpSR=0, weightMantSR=255)
    return dict(schema='nxtf-partitions-v1',
                provenance=dict(compiler='NxTF public partition compiler',
                                sdk_mapping=False, hardware_validation=False),
                layers=[dict(id='input', type='NxInputLayer', output_shape=[1], reset_mode='soft')],
                partitions=[dict(id='input:0', layer_id='input', type='NxInputLayer',
                    reset_mode='soft', compartment_parameters=params,
                    connection_parameters=conn, compartments=[
                        dict(id=0, global_id=0, role='dendrite', bias_mantissa=128, bias_exponent=6),
                        dict(id=1, global_id=1, role='soma', bias_mantissa=0, bias_exponent=0),
                        dict(id=2, global_id=-1, role='dummy', bias_mantissa=0, bias_exponent=0),
                        dict(id=3, global_id=-1, role='dummy', bias_mantissa=0, bias_exponent=0)],
                    synapse_formats=[], synapse_groups=[], input_axon_groups=[], output_axon_groups=[],
                    resource_counts={})],
                edges=[dict(source_partition='input:0', source_cx=1,
                    target_partition='input:0', target_cx=0, weight_mantissa=-255,
                    weight_exponent=0, num_weight_bits=8, sign_mode=3,
                    soft_reset=True, delay=0)])


class TestNxTFManifest(unittest.TestCase):
    def validate(self, doc, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('sanafe.nxtf'), 'NxTF import boundary missing')
        from sanafe.nxtf import validate_partition_manifest
        return validate_partition_manifest(doc, **kwargs)

    def test_compartments_and_feedback_preserved(self):
        model = self.validate(example())
        self.assertEqual(len(model['nodes']), 4)
        self.assertEqual(model['nodes'][0]['bias'], 8192)
        self.assertEqual(model['edges'][0]['weight'], -16320)
        self.assertEqual(model['soft_reset_pairs'], 1)
        self.assertFalse(model['sdk_mapping'])

    def test_reject_inconsistent_compiler_parameters(self):
        for key,value in [('weight_exponent',1),('weight_mantissa',-254)]:
            doc=example();doc['edges'][0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'disagrees'):
                self.validate(doc)

    def test_reject_unimplemented_input_and_encoding(self):
        doc=example();doc['layers'][0]['signed']=True
        with self.assertRaises(ValueError):self.validate(doc)
        doc=example();doc['layers'][0]['input_mode']=1
        with self.assertRaises(ValueError):self.validate(doc)
        doc=example();doc['partitions'][0]['connection_parameters']['synapseEncoding']='dense2'
        with self.assertRaises(ValueError):self.validate(doc)

    def test_reject_unknown_schema(self):
        doc = example(); doc['schema'] = 'loihi2'
        with self.assertRaisesRegex(ValueError, 'schema'): self.validate(doc)

    def test_reject_unsupported_dynamics(self):
        for key, value in [('compartmentVoltageDecay', 1), ('compartmentCurrentDecay', 4095),
                           ('refractoryDelay', 1), ('threshOp', 3), ('mystery', 1)]:
            doc = example(); doc['partitions'][0]['compartment_parameters'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(doc)

    def test_reject_noninteger_weight_and_delay(self):
        for key,value in [('weight_mantissa', 0.5), ('delay', 1), ('num_weight_bits', 4)]:
            doc=example(); doc['edges'][0][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(doc)

    def test_reject_missing_or_wrong_reset(self):
        doc=example(); doc['edges']=[]
        with self.assertRaisesRegex(ValueError, 'feedback'): self.validate(doc)
        doc=example(); doc['edges'][0]['source_cx']=0
        with self.assertRaises(ValueError): self.validate(doc)

    def test_reject_duplicate_compartment(self):
        doc=example(); doc['partitions'][0]['compartments'][1]['id']=0
        with self.assertRaises(ValueError): self.validate(doc)

    def test_reject_missing_pair(self):
        doc=example(); doc['partitions'][0]['compartments'][0]['role']='hard'
        with self.assertRaises(ValueError): self.validate(doc)

    def test_record_reset_approximation(self):
        doc=example(); doc['partitions'][0]['compartment_parameters']['vThMant']=512
        doc['partitions'][0]['connection_parameters'].update(weightExpSR=1)
        doc['edges'][0]['weight_exponent']=1
        model=self.validate(doc)
        self.assertEqual(model['reset_differences'], [{'partition':'input:0', 'cx':0,
                         'threshold':32768, 'feedback':32640, 'difference':128}])

    def test_input_override_keeps_compiled_graph(self):
        doc=example(); before=copy.deepcopy(doc)
        model=self.validate(doc,input_biases={'input':[255]})
        self.assertEqual(model['nodes'][0]['bias'],16320)
        self.assertEqual(doc,before)
        self.assertEqual(model['nodes'][1]['bias'],0)

    def test_reject_input_shape_and_unknown_layer(self):
        for biases in [{'input':[]}, {'input':[0.5]}, {'unknown':[1]}]:
            with self.subTest(biases=biases), self.assertRaises(ValueError):
                self.validate(example(),input_biases=biases)

    def test_reject_core_overlap_and_out_of_bounds(self):
        for placement in [{'input:0':[32,0]}, {'input:0':[0,4]}, {}, {'input:0':[0,True]}]:
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                self.validate(example(),placements=placement)

    def test_placement_dictionary_order_does_not_change_graph(self):
        doc=example()
        second=copy.deepcopy(doc['partitions'][0])
        second['id']='output:0'; second['layer_id']='output'; second['type']='NxDense'
        second['compartments'][0]['bias_mantissa']=0
        doc['layers'].append(dict(id='output',type='NxDense',reset_mode='soft'))
        doc['partitions'].append(second)
        reset=copy.deepcopy(doc['edges'][0])
        reset.update(source_partition='output:0',target_partition='output:0')
        doc['edges'].append(reset)
        forward=copy.deepcopy(reset)
        forward.update(source_partition='input:0',weight_mantissa=255,sign_mode=2,soft_reset=False)
        doc['edges'].append(forward)
        from sanafe.nxtf import map_compiled_partitions
        import sanafe
        traces=[]
        for placement in ({'input:0':[0,0],'output:0':[1,0]},
                          {'output:0':[1,0],'input:0':[0,0]}):
            net,arch,_=map_compiled_partitions(doc,placements=placement)
            chip=sanafe.SpikingChip(arch);chip.load(net)
            traces.append(chip.sim(6,potential_trace=True)['potential_trace'])
        self.assertEqual(traces[0],traces[1])
        self.assertEqual(traces[0][3][4],16320)

    def test_execute_real_pair_with_explicit_feedback(self):
        self.validate(example())
        from sanafe.nxtf import map_compiled_partitions
        import sanafe
        net, arch, report=map_compiled_partitions(example())
        chip=sanafe.SpikingChip(arch); chip.load(net)
        result=chip.sim(4, potential_trace=True,spike_trace=True)
        self.assertEqual(result['potential_trace'], [[8192,8192,0,0],[16384,0,0,0],
                                                    [8256,8256,0,0],[16448,0,0,0]])
        self.assertEqual([len(x) for x in result['spike_trace']],[0,1,0,1])
        self.assertEqual(report['placement_provenance'],'explicit SANA-FE placement')
        chip.reset()
        self.assertEqual(chip.sim(4,potential_trace=True)['potential_trace'],result['potential_trace'])


if __name__=='__main__':unittest.main()
