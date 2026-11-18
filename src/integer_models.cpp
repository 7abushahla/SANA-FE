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

void sanafe::GatedInt24IfModel::set_attribute_neuron(const size_t address,
        const std::string &name, const ModelAttribute &param)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    const auto int32_attr = [&](const int64_t lo, const int64_t hi, const char *what) {
        return static_cast<int32_t>(checked_integer(static_cast<double>(param), lo, hi, what));
    };
    if (name == "currents")
    {
        std::vector<int32_t> values;
        for (const auto &value : std::get<std::vector<ModelAttribute>>(param.value))
        {
            values.push_back(static_cast<int32_t>(checked_integer(
                    static_cast<double>(value), voltage_min, voltage_max,
                    "IF signed 24-bit external current")));
        }
        state.currents = std::move(values);
        state.cursor = 0;
    }
    else if (name == "graded_values")
    {
        std::vector<int32_t> values;
        for (const auto &value : std::get<std::vector<ModelAttribute>>(param.value))
        {
            values.push_back(static_cast<int32_t>(checked_integer(
                    static_cast<double>(value), std::numeric_limits<int32_t>::min(),
                    std::numeric_limits<int32_t>::max(), "IF signed 32-bit graded value")));
        }
        state.graded_values = std::move(values);
        state.graded_cursor = 0;
        state.graded_source = true;
    }
    else if (name == "threshold")
    {
        const int32_t threshold = int32_attr(1, voltage_max, "IF threshold");
        if (threshold % 2 != 0)
        {
            throw std::invalid_argument("IF threshold must be even");
        }
        state.threshold = threshold;
    }
    else if (name == "initial_voltage")
    {
        state.initial = int32_attr(voltage_min, voltage_max, "IF signed 24-bit initial voltage");
        state.voltage = state.initial;
    }
    else if (name == "bias_at_start")
    {
        state.bias_at_start = int32_attr(voltage_min, voltage_max, "IF signed 24-bit window bias");
    }
    else if (name == "bias_per_check")
    {
        state.bias_per_check = static_cast<int16_t>(
                int32_attr(current_min, current_max, "IF signed 16-bit bias per check"));
    }
    else if (name == "window_start")
    {
        state.window_start = int32_attr(0, std::numeric_limits<int32_t>::max() - 1, "IF window_start");
    }
    else if (name == "window_stop")
    {
        state.window_stop = int32_attr(1, std::numeric_limits<int32_t>::max(), "IF window_stop");
    }
    else if (name == "stride")
    {
        state.stride = int32_attr(1, std::numeric_limits<int32_t>::max(), "IF stride");
    }
}

sanafe::PipelineResult sanafe::GatedInt24IfModel::update(const size_t address,
        const std::optional<double> current, const long int /*timestep*/)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    if (state.graded_source)
    {
        // A graded source sends one value per update and integrates nothing. It ignores the
        // threshold, the window, both biases and the external currents, and its voltage stays
        // at its initial value.
        if (current.has_value() && current.value() != 0.0)
        {
            throw std::invalid_argument("A graded source takes no synaptic input");
        }
        const int32_t value = state.graded_cursor < state.graded_values.size()
                ? state.graded_values.at(state.graded_cursor) : 0;
        if (state.graded_cursor < state.graded_values.size())
        {
            ++state.graded_cursor;
        }
        if (value == 0)
        {
            return {std::nullopt, updated, std::nullopt, std::nullopt};
        }
        return {static_cast<double>(value), fired, std::nullopt, std::nullopt};
    }
    if (state.window_stop <= state.window_start)
    {
        throw std::invalid_argument("IF window requires window_start < window_stop");
    }
    const int64_t external = state.cursor < state.currents.size()
            ? state.currents.at(state.cursor) : 0;
    if (state.cursor < state.currents.size())
    {
        ++state.cursor;
    }
    // Synaptic and external current together form one signed 24-bit input per update.
    const int64_t input = checked_integer(current.value_or(0.0) + static_cast<double>(external),
            voltage_min, voltage_max, "IF signed 24-bit input current");
    const auto u = static_cast<int64_t>(state.updates_seen++);
    int64_t v = static_cast<int64_t>(state.voltage) + input;
    if (u == state.window_start)
    {
        v += state.bias_at_start;
    }
    const bool check = u >= state.window_start && u < state.window_stop &&
            (u - state.window_start) % state.stride == 0;
    if (check)
    {
        v += state.bias_per_check;
    }
    state.voltage = static_cast<int32_t>(std::clamp(v, voltage_min, voltage_max));
    const bool spike = check && state.voltage >= state.threshold;
    if (spike)
    {
        state.voltage -= state.threshold;
    }
    return {std::nullopt, spike ? fired : updated, std::nullopt, std::nullopt};
}

void sanafe::GatedInt24IfModel::reset()
{
    for (State &state : states)
    {
        state.voltage = state.initial;
        state.cursor = 0;
        state.updates_seen = 0;
        state.graded_cursor = 0;
    }
}

namespace
{
// Lava's fixed-point LIF registers: 24-bit u and v, 12-bit decays, 6-bit alignment.
constexpr int64_t lif_uv_bound = 1LL << 23;
constexpr int64_t lif_decay_unity = 1LL << 12;
constexpr int64_t lif_alignment = 1LL << 6;
// Largest activation magnitude, 2^18 + 32, for which Lava's single wrap by 2^24 keeps the
// current in (-2^23, 2^23] from every reachable current and decay (the decay keeps at most
// 4095/4096 of the current). Lava shifts any signed 32-bit activation without a check. Beyond
// this bound its current leaves 24 bits and can later overflow its 32-bit arithmetic.
constexpr int64_t lif_activation_bound = (1LL << 18) + 32;

// sign(p) * (|p| >> 12) on a 64-bit product, as in AbstractPyLifModelFixed.subthr_dynamics.
int64_t lif_decay(const int64_t value, const int64_t keep)
{
    const int64_t product = value * keep;
    return product >= 0 ? product / lif_decay_unity : -((-product) / lif_decay_unity);
}
}

void sanafe::CubaLifInt24Model::set_attribute_neuron(const size_t address,
        const std::string &name, const ModelAttribute &param)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    const auto int32_attr = [&](const int64_t lo, const int64_t hi, const char *what) {
        return static_cast<int32_t>(checked_integer(static_cast<double>(param), lo, hi, what));
    };
    if (name == "currents")
    {
        std::vector<int32_t> values;
        for (const auto &value : std::get<std::vector<ModelAttribute>>(param.value))
        {
            values.push_back(static_cast<int32_t>(checked_integer(
                    static_cast<double>(value), -lif_activation_bound, lif_activation_bound,
                    "LIF external activation")));
        }
        state.currents = std::move(values);
        state.cursor = 0;
    }
    else if (name == "du")
    {
        state.du = int32_attr(0, lif_decay_unity - 1, "LIF unsigned 12-bit du");
    }
    else if (name == "dv")
    {
        state.dv = int32_attr(0, lif_decay_unity - 1, "LIF unsigned 12-bit dv");
    }
    else if (name == "vth")
    {
        state.vth = int32_attr(0, (1LL << 17) - 1, "LIF unsigned 17-bit vth");
    }
    else if (name == "bias_mant")
    {
        state.bias_mant = int32_attr(-(1LL << 12), (1LL << 12) - 1, "LIF signed 13-bit bias_mant");
    }
    else if (name == "bias_exp")
    {
        state.bias_exp = int32_attr(0, 7, "LIF unsigned 3-bit bias_exp");
    }
    else if (name == "reset_interval")
    {
        state.reset_interval = int32_attr(0, std::numeric_limits<int32_t>::max(), "LIF reset_interval");
    }
    else if (name == "reset_offset")
    {
        state.reset_offset = int32_attr(0, std::numeric_limits<int32_t>::max(), "LIF reset_offset");
    }
    else if (name == "initial_v")
    {
        state.initial_v = int32_attr(voltage_min, voltage_max, "LIF signed 24-bit initial_v");
        state.v = state.initial_v;
    }
}

sanafe::PipelineResult sanafe::CubaLifInt24Model::update(const size_t address,
        const std::optional<double> current, const long int /*timestep*/)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    const int64_t external = state.cursor < state.currents.size()
            ? state.currents.at(state.cursor) : 0;
    if (state.cursor < state.currents.size())
    {
        ++state.cursor;
    }
    // Synaptic input and external activation together form Lava's a_in.
    const int64_t activation = checked_integer(current.value_or(0.0) + static_cast<double>(external),
            -lif_activation_bound, lif_activation_bound, "LIF input activation");
    // Lava's time_step is one-based and persists across chunked runs until reset().
    const auto t = static_cast<int64_t>(++state.updates_seen);
    if (state.reset_interval > 0 &&
            t % state.reset_interval == state.reset_offset % state.reset_interval)
    {
        state.u = 0;
        state.v = 0;
    }
    // Current: decay by (du + 1) / 4096, add the aligned input, wrap into 24 bits.
    const int64_t x = lif_decay(state.u, lif_decay_unity - (state.du + 1)) +
            activation * lif_alignment;
    const int64_t w = x > lif_uv_bound ? x - 2 * lif_uv_bound : x;
    state.u = static_cast<int32_t>(w <= -lif_uv_bound ? x + 2 * lif_uv_bound : w);
    // Voltage: decay by dv / 4096, add current and bias, clip to the symmetric 24-bit range.
    const int64_t bias = static_cast<int64_t>(state.bias_mant) * (1LL << state.bias_exp);
    const int64_t v = lif_decay(state.v, lif_decay_unity - state.dv) + state.u + bias;
    state.v = static_cast<int32_t>(std::clamp(v, -lif_uv_bound + 1, lif_uv_bound - 1));
    const bool spike = state.v > static_cast<int64_t>(state.vth) * lif_alignment;
    if (spike)
    {
        state.v = 0;
    }
    return {std::nullopt, spike ? fired : updated, std::nullopt, std::nullopt};
}

void sanafe::CubaLifInt24Model::reset()
{
    for (State &state : states)
    {
        state.u = 0;
        state.v = state.initial_v;
        state.cursor = 0;
        state.updates_seen = 0;
    }
}
