#include <gtest/gtest.h>

#include "arch.hpp"
#include "core.hpp"
#include "mapped.hpp"
#include "network.hpp"
#include "pipeline.hpp"
#include "timestep.hpp"

namespace
{
class MetricUnit final : public sanafe::PipelineUnit
{
public:
    MetricUnit() : PipelineUnit(sanafe::implements_soma)
    {
        process_input_fn = [](sanafe::Timestep &, sanafe::MappedNeuron &,
                               std::optional<sanafe::MappedConnection *>,
                               const sanafe::PipelineResult &) {
            sanafe::PipelineResult result;
            result.status = sanafe::updated;
            result.energy = 2.0;
            result.latency = 7.0;
            return result;
        };
        process_output_fn = [](sanafe::MappedNeuron &,
                                std::optional<sanafe::MappedConnection *>,
                                sanafe::PipelineResult &) {};
    }
    void set_attribute_hw(const std::string &, const sanafe::ModelAttribute &) override {}
    void set_attribute_neuron(size_t, const std::string &, const sanafe::ModelAttribute &) override {}
    void set_attribute_edge(size_t, const std::string &, const sanafe::ModelAttribute &) override {}
    void reset() override {}
};
}

TEST(PipelineMetrics, TracksLatencyIndependentlyOfEnergy)
{
    sanafe::CorePipelineConfiguration pipeline;
    sanafe::CoreConfiguration configuration("test", sanafe::CoreAddress{}, pipeline);
    sanafe::Core core(configuration);
    sanafe::SpikingNetwork network("test");
    sanafe::NeuronConfiguration neuron_configuration;
    auto &group = network.create_neuron_group("layer", 1, neuron_configuration);
    MetricUnit unit;
    sanafe::MappedNeuron neuron(0, group.neurons.at(0), 0, &core, &unit, nullptr, nullptr);
    sanafe::Timestep timestep(1);
    const sanafe::PipelineResult empty;
    unit.process(timestep, neuron, std::nullopt, empty);
    unit.process(timestep, neuron, std::nullopt, empty);
    EXPECT_DOUBLE_EQ(unit.energy, 4.0);
    EXPECT_DOUBLE_EQ(unit.latency, 14.0);
}
