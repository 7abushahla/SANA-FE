// Integer numerical reference models. No hardware ISA fidelity is implied.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "models.hpp"

namespace
{
constexpr int64_t voltage_min = -(1LL << 23);
constexpr int64_t voltage_max = (1LL << 23) - 1;
constexpr int64_t current_min = -(1LL << 15);
constexpr int64_t current_max = (1LL << 15) - 1;

int64_t checked_integer(const double value, const int64_t minimum,
        const int64_t maximum, const std::string &description)
{
    if (!std::isfinite(value) || std::trunc(value) != value ||
            value < static_cast<double>(minimum) ||
            value > static_cast<double>(maximum))
    {
        throw std::invalid_argument(description +
                " must be a finite integer in [" + std::to_string(minimum) +
                ", " + std::to_string(maximum) + "]");
    }
    return static_cast<int64_t>(value);
}
}

void sanafe::IntegerCurrentBasedSynapseModel::set_attribute_edge(
        const size_t address, const std::string &name, const ModelAttribute &param)
{
    if (name == "weight" || name == "w")
    {
        const auto weight = checked_integer(static_cast<double>(param),
                current_min, current_max, "Integer synapse signed 16-bit weight");
        if (weights.size() <= address)
        {
            weights.resize(address + 1);
        }
        weights.at(address) = static_cast<int16_t>(weight);
    }
}

sanafe::PipelineResult sanafe::IntegerCurrentBasedSynapseModel::update(
        const size_t address, const bool read, const long int /*timestep*/)
{
    PipelineResult result;
    result.current = read ? static_cast<double>(weights.at(address)) : 0.0;
    return result;
}

sanafe::PipelineResult sanafe::IntegerAccumulatorModel::update(
        const size_t address, const std::optional<double> current,
        const std::optional<size_t> /*synapse_address*/, const long int timestep)
{
    if (charges.size() <= address)
    {
        charges.resize(address + 1, std::nullopt);
        timestamps.resize(address + 1, 0L);
    }
    if (timestamps.at(address) < timestep)
    {
        charges.at(address) = 0;
        timestamps.at(address) = timestep;
    }
    if (current.has_value())
    {
        const int64_t input = checked_integer(current.value(),
                std::numeric_limits<int32_t>::min(),
                std::numeric_limits<int32_t>::max(), "Integer accumulator input");
        const int64_t sum = charges.at(address).value_or(0) + input;
        if (sum < std::numeric_limits<int32_t>::min() ||
                sum > std::numeric_limits<int32_t>::max())
        {
            throw std::overflow_error("Integer accumulator sum exceeds signed 32-bit range");
        }
        charges.at(address) = sum;
    }
    PipelineResult result;
    if (charges.at(address).has_value())
    {
        // Every signed 32-bit integer is represented exactly by double.
        result.current = static_cast<double>(charges.at(address).value());
    }
    return result;
}

void sanafe::IntegerAccumulatorModel::reset()
{
    charges.assign(charges.size(), std::nullopt);
    timestamps.assign(timestamps.size(), 0L);
}

void sanafe::Int24IfModel::set_attribute_neuron(const size_t address,
        const std::string &name, const ModelAttribute &param)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    if (name == "currents")
    {
        std::vector<int16_t> values;
        for (const auto &value : std::get<std::vector<ModelAttribute>>(param.value))
        {
            values.push_back(static_cast<int16_t>(checked_integer(
                    static_cast<double>(value), current_min, current_max,
                    "IF signed 16-bit external current")));
        }
        state.currents = std::move(values);
        state.cursor = 0;
    }
    else if (name == "threshold")
    {
        const int64_t threshold = checked_integer(static_cast<double>(param),
                1, voltage_max, "IF threshold");
        if (threshold % 2 != 0)
        {
            throw std::invalid_argument("IF threshold must be even");
        }
        state.threshold = static_cast<int32_t>(threshold);
    }
    else if (name == "initial_voltage")
    {
        state.initial = static_cast<int32_t>(checked_integer(
                static_cast<double>(param), voltage_min, voltage_max,
                "IF signed 24-bit initial voltage"));
        state.voltage = state.initial;
    }
    else if (name == "bias")
    {
        state.bias = static_cast<int16_t>(checked_integer(
                static_cast<double>(param), current_min, current_max,
                "IF signed 16-bit bias"));
    }
    else if (name == "valid_start")
    {
        state.valid_start = static_cast<int32_t>(checked_integer(
                static_cast<double>(param), 0,
                std::numeric_limits<int32_t>::max() - 1,
                "IF valid_start"));
    }
    else if (name == "valid_stop")
    {
        state.valid_stop = static_cast<int32_t>(checked_integer(
                static_cast<double>(param), 1,
                std::numeric_limits<int32_t>::max(),
                "IF valid_stop"));
    }
}

sanafe::PipelineResult sanafe::Int24IfModel::update(const size_t address,
        const std::optional<double> current, const long int /*timestep*/)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    const int64_t incoming = checked_integer(current.value_or(0.0),
            current_min, current_max, "IF signed 16-bit input current");
    if (state.valid_stop <= state.valid_start)
    {
        throw std::invalid_argument("IF valid update window requires start < stop");
    }
    // chip.sim() may be called in chunks. Its supplied timestep restarts for
    // each call, whereas the IF state persists. Track this neuron's absolute
    // update index until reset instead of using the per-call timestep.
    const size_t update_index = state.updates_seen++;
    if (update_index < static_cast<size_t>(state.valid_start) ||
            update_index >= static_cast<size_t>(state.valid_stop))
    {
        if (state.cursor < state.currents.size())
        {
            ++state.cursor;
        }
        return {std::nullopt, idle, std::nullopt, std::nullopt};
    }
    const int64_t external = state.cursor < state.currents.size()
            ? state.currents.at(state.cursor) : 0;
    // The stream and synaptic current jointly form the signed 16-bit input.
    const int64_t input = checked_integer(static_cast<double>(incoming + external),
            current_min, current_max, "IF signed 16-bit combined input current");
    const int64_t integrated = static_cast<int64_t>(state.voltage) + input + state.bias;
    state.voltage = static_cast<int32_t>(std::clamp(integrated, voltage_min, voltage_max));
    const bool spike = state.voltage >= state.threshold;
    if (spike)
    {
        state.voltage -= state.threshold;
    }
    if (state.cursor < state.currents.size())
    {
        ++state.cursor;
    }
    return {std::nullopt, spike ? fired : updated, std::nullopt, std::nullopt};
}

void sanafe::Int24IfModel::reset()
{
    for (State &state : states)
    {
        state.voltage = state.initial;
        state.cursor = 0;
        state.updates_seen = 0;
    }
}
