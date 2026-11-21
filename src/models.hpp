// Copyright (c) 2026 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
// models.hpp
#ifndef MODELS_HEADER_INCLUDED_
#define MODELS_HEADER_INCLUDED_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory> // For shared_ptr<T>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arch.hpp"
#include "fwd.hpp"
#include "pipeline.hpp"

namespace sanafe
{
constexpr int default_weight_bits = 8; // Based on real-world H/W e.g., Loihi
constexpr int loihi_max_compartments{1024};

class CurrentBasedSynapseModel : public SynapseUnit
{
public:
    CurrentBasedSynapseModel() { register_attributes(current_based_synapse_attributes); }
    CurrentBasedSynapseModel(const CurrentBasedSynapseModel &copy) = default;
    CurrentBasedSynapseModel(CurrentBasedSynapseModel &&other) = default;
    ~CurrentBasedSynapseModel() override = default;
    CurrentBasedSynapseModel &operator=(const CurrentBasedSynapseModel &other) = delete;
    CurrentBasedSynapseModel &operator=(CurrentBasedSynapseModel &&other) = delete;

    PipelineResult update(size_t synapse_address, bool read, long int simulation_time) override;
    void set_attribute_hw(const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_edge(size_t synapse_address, const std::string &attribute_name, const ModelAttribute &param) override;
    void reset() override;
    static inline const std::unordered_map<std::string, std::string> current_based_synapse_attributes{
        {"weight", "(float) Synaptic weight associated with connection."},
        {"w", "(float) Synaptic weight associated with connection."},
        {"delay", "(float) Time-steps that a spike is delayed."},
        {"d", "(float) Time-steps that a spike is delayed."},
    };

private:
    std::vector<double> weights;
    double min_synaptic_resolution{0.0};
    int weight_bits{default_weight_bits};
};

class AccumulatorModel : public DendriteUnit
{
public:
    AccumulatorModel()
    {
        register_attributes(accumulator_attributes);
    }
    AccumulatorModel(const AccumulatorModel &copy) = default;
    AccumulatorModel(AccumulatorModel &&other) = default;
    ~AccumulatorModel() override = default;
    AccumulatorModel &operator=(const AccumulatorModel &other) = delete;
    AccumulatorModel &operator=(AccumulatorModel &&other) = delete;

    PipelineResult update(size_t neuron_address, std::optional<double> current, std::optional<size_t> synapse_address, long int simulation_time) override;
    void reset() override
    {
        accumulated_charges.assign(accumulated_charges.size(), std::nullopt);
        timesteps_simulated.assign(timesteps_simulated.size(), 0L);
    }
    void set_attribute_hw(const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_neuron(size_t /*neuron_address*/, const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_edge(size_t /*synapse_address*/, const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    // Technically soma attributes, but due to common usage, suppress warnings
    //  for these in the dendrite H/W too
    static inline const std::set<std::string> accumulator_attributes{
            "reset_mode",
            "reverse_reset_mode",
            "reset",
            "reverse_reset",
            "bias",
            "threshold",
            "reverse_threshold",
            "leak_decay",
            "noise",
            "weight",
            "w",
            "latency",
    };

private:
    std::vector<std::optional<double>> accumulated_charges;
    std::vector<long int> timesteps_simulated;
};

class AccumulatorWithDelayModel : public DendriteUnit
{
public:
    AccumulatorWithDelayModel()
    {
        next_accumulated_charges.resize(max_delay + 1UL);
        register_attributes(accumulator_attributes);
    }
    AccumulatorWithDelayModel(const AccumulatorWithDelayModel &copy) = default;
    AccumulatorWithDelayModel(AccumulatorWithDelayModel &&other) = default;
    ~AccumulatorWithDelayModel() override = default;
    AccumulatorWithDelayModel &operator=(const AccumulatorWithDelayModel &other) = delete;
    AccumulatorWithDelayModel &operator=(AccumulatorWithDelayModel &&other) = delete;

    PipelineResult update(size_t neuron_address, std::optional<double> current, std::optional<size_t> synapse_address, long int simulation_time) override;
    void reset() override
    {
        accumulated_charges.assign(accumulated_charges.size(), std::nullopt);
        timesteps_simulated.assign(timesteps_simulated.size(), 0L);
        for (auto &accumulator : next_accumulated_charges)
        {
            accumulator.assign(accumulated_charges.size(), std::nullopt);
        }
    }
    void set_attribute_hw(const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_neuron(size_t /*neuron_address*/, const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_edge(size_t synapse_address, const std::string &attribute_name, const ModelAttribute &param) override;
    void track_connection(size_t synapse_address, size_t /*src_neuron_id*/, size_t /*dest_neuron_id*/) override;

    static inline const std::set<std::string> accumulator_attributes{
            "reset_mode",
            "reverse_reset_mode",
            "reset",
            "reverse_reset",
            "bias",
            "threshold",
            "reverse_threshold",
            "leak_decay",
            "noise",
            "weight",
            "w",
            "latency",
            "delay",
            "d",
    };

private:
    // Technically soma attributes, but due to common usage, suppress warnings
    //  for these in the dendrite H/W too
    const size_t max_delay{5UL};
    std::vector<std::optional<double>> accumulated_charges;
    std::vector<std::vector<std::optional<double>>> next_accumulated_charges;
    std::vector<long int> timesteps_simulated;
    std::vector<size_t> delays;
};

class MultiTapModel1D : public DendriteUnit
{
public:
    MultiTapModel1D() = default;
    MultiTapModel1D(const MultiTapModel1D &copy) = default;
    MultiTapModel1D(MultiTapModel1D &&other) = default;
    ~MultiTapModel1D() override = default;
    MultiTapModel1D &operator=(const MultiTapModel1D &other) = delete;
    MultiTapModel1D &operator=(MultiTapModel1D &&other) = delete;

    PipelineResult update(size_t neuron_address, std::optional<double> current, std::optional<size_t> synapse_address, long int simulation_time) override;
    void set_attribute_hw(const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_neuron(size_t neuron_address, const std::string &attribute_name, const ModelAttribute &param) override;
    void set_attribute_edge(size_t synapse_address, const std::string &attribute_name, const ModelAttribute &param) override;
    void reset() override;
    static inline const std::unordered_map<std::string, std::string> multitap_attributes{
            {"taps", "(int) Number of N dendritic 'taps' or compartments."},
            {"time_constants", "(list[float]) List of RC line time constants, one per tap."},
            {"space_constants", "(list[float]) List of RC line space constants, N-1 in total."}
    };

private:
    // Modeling a 1D dendrite with taps
    std::vector<double> tap_voltages{std::vector<double>(1, 0.0)};
    std::vector<double> next_voltages{std::vector<double>(1, 0.0)};
    std::vector<double> space_constants{std::vector<double>(0)};
    std::vector<double> time_constants{std::vector<double>(1, 0.0)};
    std::vector<int> synapse_to_tap;
    long int timesteps_simulated{0L};

    void calculate_next_state();
    void input_current(double current, std::optional<size_t> synapse_address);
    void print_taps();
};

class LoihiLifModel : public SomaUnit
{
public:
    LoihiLifModel() { register_attributes(loihi_lif_attributes); }
    LoihiLifModel(const LoihiLifModel &copy) = delete;
    LoihiLifModel(LoihiLifModel &&other) = delete;
    ~LoihiLifModel() override = default;
    LoihiLifModel &operator=(const LoihiLifModel &other) = delete;
    LoihiLifModel &operator=(LoihiLifModel &&other) = delete;

    void set_attribute_hw(const std::string &attribute_name, const ModelAttribute &param) override;
    void set_attribute_neuron(size_t neuron_address, const std::string &attribute_name, const ModelAttribute &param) override;
    PipelineResult update(size_t neuron_address, std::optional<double> current_in, long int simulation_time) override;
    void reset() override;
    double get_potential(const size_t neuron_address) override
    {
        return compartments[neuron_address].potential;
    }
    std::map<std::string, double> get_neuron_traces(size_t neuron_address) override;
    std::uniform_real_distribution<double> uniform_distribution{0.0, 1.0}; // TODO: hack
    std::mt19937 gen{1}; // TODO: hack

    struct LoihiCompartment
    {
        double bias{0.0};
        std::vector<double> biases;
        bool force_update_every_timestep{false};
        double input_current{0.0};
        double input_decay{0.0};
        double leak_decay{1.0};
        bool log_current{false};
        double potential{0.0};
        int refractory_delay{0};
        int refractory_count{0};
        double reset{0.0};
        int reset_mode{neuron_reset_hard};
        double reverse_reset{0.0};
        int reverse_reset_mode{neuron_no_reset};
        double reverse_threshold{0.0};
        double threshold{0.0};
        long int timesteps_simulated{0L};
    };

    enum NoiseType : uint8_t
    {
        noise_none = 0U,
        noise_file_stream = 1U,
    };

    static inline const std::unordered_map<std::string, std::string> loihi_lif_attributes{
            {"bias", "Bias current applied every step: v[t+1] = bias + v[t]*leak_decay + u[t]"},
            {"force_update", "(bool) Force soma to update every step, regardless of inputs."},
            {"force_update_every_timestep", "(bool) Force soma to update every step, regardless of inputs."},
            {"force_potential", ""},
            {"leak_decay", "(float) Decay term applied every step: v[t+1] = bias + v[t]*leak_decay + u[t]"},
            {"log_u", "(bool) Record input current (u) for Loihi soma."},
            {"noise", "(str) Noise source. Only file-based noise stream supported"},
            {"noise_bits", "(int) The number of noise bits simulated."},
            {"refractory_delay", "(int) The number of refractory steps after a spike, default=0."},
            {"reset_mode", "(str) The type of reset to apply on spikes [none/soft/hard/saturate]. Default=hard"},
            {"reverse_reset_mode", "(str) The type of reset to apply on negative/reverse spikes [none/soft/hard/saturate]. Default=None"},
            {"reset", "(float) The potential to reset to after a spike. Default=0.0"},
            {"reverse_reset", "(float) The potential to reset to after a reverse spike."},
            {"reverse_threshold", "(float) Fires when V < reverse_threshold, then V <- f(reverse_reset_mode, reverse_reset)."},
            {"threshold", "(float) Fires when V > threshold, then V <- f(reset_mode, reset)."},
    };

private:
    std::vector<LoihiCompartment> compartments{loihi_max_compartments};
    NoiseType noise_type{noise_none};
    std::ifstream noise_stream;
    long int sign_mask{0x100}; // One more than the max noise bits (=8)
    long int random_mask{0x7f};
    int noise_bits{7};

    static void loihi_leak(LoihiCompartment &cx);
    static void loihi_quantize(LoihiCompartment &cx);
    static bool loihi_threshold_and_reset(LoihiCompartment &cx);

    double loihi_generate_noise();
    int loihi_read_noise_stream();
    void loihi_reset_noise_stream();
};

constexpr int truenorth_max_neurons{4096};
class TrueNorthModel : public SomaUnit
{
public:
    TrueNorthModel() { register_attributes(truenorth_attributes); }
    TrueNorthModel(const TrueNorthModel &copy) = default;
    TrueNorthModel(TrueNorthModel &&other) = default;
    ~TrueNorthModel() override = default;
    TrueNorthModel &operator=(const TrueNorthModel &other) = delete;
    TrueNorthModel &operator=(TrueNorthModel &&other) = delete;

    void set_attribute_hw(const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_neuron(size_t neuron_address,
            const std::string &attribute_name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t neuron_address,
            std::optional<double> current_in, long int simulation_time) override;
    void reset() override;
    double get_potential(const size_t neuron_address) override
    {
        return neurons[neuron_address].potential;
    }

    struct TrueNorthNeuron
    {
        bool force_update{false};
        unsigned int random_range_mask{0U};
        int reset_mode{neuron_reset_hard};
        int reverse_reset_mode{neuron_no_reset};
        bool leak_towards_zero{true};
        // int noise_type;
        double potential{0.0};
        double leak{0.0}; // Default is no leak (potential decay)
        double bias{0.0};
        double threshold{0.0};
        double reverse_threshold{0.0};
        double reset{0.0};
        double reverse_reset{0.0};
    };

    static inline const std::unordered_map<std::string, std::string> truenorth_attributes{
        {"reset", "(float) The potential to reset to after a spike. Default=0.0"},
        {"reverse_reset", "(float) The potential to reset to after a reverse spike."},
        {"reset_mode", "(str) The type of reset to apply on spikes [none/soft/hard/saturate]. Default=hard"},
        {"reverse_reset_mode", "(str) The type of reset to apply on negative/reverse spikes [none/soft/hard/saturate]. Default=None"},
        {"bias", "Additive bias current applied every step."},
        {"threshold", "(float) The potential at which a spike is triggered."},
        {"reverse_threshold", "(float) The potential at which a reverse spike is triggered."},
        {"leak", "(float) Subtractive leak term applied every step"},
        {"leak_towards_zero", "(bool) Leak towards zero if true, leak away otherwise."},
        {"random_mask", "(int) Positive mask to apply to randomly generated noise."},
    };

private:
    std::vector<TrueNorthNeuron> neurons{truenorth_max_neurons};

    static void truenorth_leak(TrueNorthNeuron &n);
    static bool truenorth_threshold_and_reset(TrueNorthNeuron &n);
};

// Integer numerical reference models. These do not implement a chip ISA.
class IntegerCurrentBasedSynapseModel : public SynapseUnit
{
public:
    IntegerCurrentBasedSynapseModel() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_edge(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, bool read, long int timestep) override;
    void reset() override {}
    static inline const std::unordered_map<std::string, std::string> attributes{
        {"weight", "(int) Effective signed 16-bit synaptic weight."},
        {"w", "(int) Effective signed 16-bit synaptic weight."},
    };
private:
    std::vector<int16_t> weights;
};

class IntegerAccumulatorModel : public DendriteUnit
{
public:
    IntegerAccumulatorModel() { register_attributes(AccumulatorModel::accumulator_attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t, const std::string &, const ModelAttribute &) override {}
    void set_attribute_edge(size_t, const std::string &, const ModelAttribute &) override {}
    PipelineResult update(size_t address, std::optional<double> current,
            std::optional<size_t> synapse_address, long int timestep) override;
    void reset() override;
private:
    std::vector<std::optional<int64_t>> charges;
    std::vector<long int> timestamps;
};

class Int24IfModel : public SomaUnit
{
public:
    Int24IfModel() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override { return states.at(address).voltage; }
    static inline const std::unordered_map<std::string, std::string> attributes{
        {"threshold", "(int) Positive even signed 24-bit threshold; comparison uses >=."},
        {"initial_voltage", "(int) Signed 24-bit initial voltage, restored on reset."},
        {"bias", "(int) Signed 16-bit constant drive added every update."},
        {"currents", "(list[int]) Signed 16-bit external current per update, then zero."},
        {"valid_start", "(int) Inclusive zero-based update index for candidate IF scheduling."},
        {"valid_stop", "(int) Exclusive zero-based update index for candidate IF scheduling."},
    };
private:
    struct State
    {
        int32_t threshold{2};
        int32_t initial{0};
        int32_t voltage{0};
        int16_t bias{0};
        int32_t valid_start{0};
        int32_t valid_stop{std::numeric_limits<int32_t>::max()};
        std::vector<int16_t> currents;
        size_t cursor{0};
        size_t updates_seen{0};
    };
    std::vector<State> states;
};

// Raised only by the strict mode of GatedInt24IfModel, when an input or a membrane leaves the
// signed 24-bit range. Python sees it as OverflowError.
class StrictRangeError : public std::overflow_error
{
public:
    using std::overflow_error::overflow_error;
};

// Loihi 2 route candidate program: integrates every update, adds its window bias once and its
// per-check bias at each check, and compares the threshold only on strided steps of its window.
// A modeled neuron program with the semantics of an integer block rule, not Intel microcode.
class GatedInt24IfModel : public SomaUnit
{
public:
    GatedInt24IfModel() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override { return states.at(address).voltage; }
    // The extremes of a strict neuron that is not a graded source and the held current of a
    // neuron with hold_input, and nothing otherwise.
    std::map<std::string, double> get_neuron_traces(size_t address) override;
    // Whether the neuron at this address holds graded_values. Not virtual, so the interface
    // that plugins compile against does not change.
    bool is_graded_source(const size_t address) const
    {
        return address < states.size() && states.at(address).graded_source;
    }
    // Whether the neuron at this address is in strict mode. Not virtual, for the same reason.
    bool is_strict(const size_t address) const
    {
        return address < states.size() && states.at(address).strict;
    }
    // Whether the neuron at this address holds its input current. Not virtual, for the same
    // reason.
    bool holds_input(const size_t address) const
    {
        return address < states.size() && states.at(address).hold_input;
    }
    static inline const std::unordered_map<std::string, std::string> attributes{
        {"threshold", "(int) Positive even signed 24-bit threshold; comparison uses >=."},
        {"initial_voltage", "(int) Signed 24-bit initial voltage, restored on reset."},
        {"bias_at_start", "(int) Signed 24-bit drive added once, at the first update of the window."},
        {"bias_per_check", "(int) Signed 16-bit drive added at every threshold check."},
        {"window_start", "(int) Zero-based update index of the first threshold check."},
        {"window_stop", "(int) Exclusive zero-based update index ending the window."},
        {"stride", "(int) Check every stride-th update of the window, from its start."},
        {"currents", "(list[int]) Signed 24-bit external current per update, then zero."},
        {"graded_values", "(list[int]) Signed 32-bit value sent per update, then zero. Sets the neuron as a graded source. Refused where the time-step buffer is inside the soma unit or before the axon-out unit, since the soma runs on the message path there."},
        {"strict", "(bool) Stop the run when the input or the membrane leaves signed 24 bits, and record their extremes as neuron traces."},
        {"hold_input", "(bool) Hold the input current. Input is added to a current register, which drives the membrane on every update before window_stop."},
    };
private:
    struct State
    {
        int32_t threshold{2};
        int32_t initial{0};
        int32_t voltage{0};
        int32_t bias_at_start{0};
        int16_t bias_per_check{0};
        int32_t window_start{0};
        int32_t window_stop{std::numeric_limits<int32_t>::max()};
        int32_t stride{1};
        std::vector<int32_t> currents;
        size_t cursor{0};
        size_t updates_seen{0};
        // A graded source sends graded_values[u] at its update u, then zero. It keeps its own
        // value cursor and never reaches the IF path, so it never advances updates_seen or the
        // cursor of currents.
        bool graded_source{false};
        std::vector<int32_t> graded_values;
        size_t graded_cursor{0};
        // In strict mode an input or a membrane beyond signed 24 bits stops the run with a
        // StrictRangeError, and the running extremes of both since the last reset are kept.
        // The membrane is the voltage after input and biases, before the clamp and the
        // threshold subtraction. Each extreme reads 0 until the first update records it.
        bool strict{false};
        bool extremes_recorded{false};
        int32_t input_min{0};
        int32_t input_max{0};
        int32_t membrane_min{0};
        int32_t membrane_max{0};
        // With hold_input, at each update u < window_stop the input is added to held_current,
        // which then drives the membrane in place of the input. From window_stop on the
        // register is left as it is and the input drives the membrane directly. The register
        // saturates at signed 24 bits, or stops the run in strict mode, and reset clears it.
        // A value held from its arrival also drives the updates before window_start, so the
        // held delivery equals the delivery at every update only for a value that arrives at
        // window_start.
        bool hold_input{false};
        int32_t held_current{0};
    };
    std::vector<State> states;
};

// Current-based LIF with the update rule of Lava's bit-accurate fixed-point LIF
// (lava/proc/lif/models.py: AbstractPyLifModelFixed.subthr_dynamics, PyLifModelBitAcc,
// PyLifResetModelBitAcc): 12-bit decays, 6-bit MSB alignment of input and threshold,
// 24-bit wrapped current, clipped voltage, strict threshold, reset to zero, periodic reset.
// For published LIF networks only. It is not the IF neuron of the integer models above.
class CubaLifInt24Model : public SomaUnit
{
public:
    CubaLifInt24Model() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override { return states.at(address).v; }
    static inline const std::unordered_map<std::string, std::string> attributes{
        {"du", "(int) Unsigned 12-bit current decay register. The current u keeps (4096 - (du + 1)) / 4096."},
        {"dv", "(int) Unsigned 12-bit voltage decay register. The voltage v keeps (4096 - dv) / 4096."},
        {"vth", "(int) Unsigned 17-bit threshold, shifted left by 6. The comparison uses >."},
        {"bias_mant", "(int) Signed 13-bit bias mantissa."},
        {"bias_exp", "(int) Unsigned 3-bit bias exponent. The bias is bias_mant << bias_exp."},
        {"reset_interval", "(int) Period in updates of the u and v reset. A value of 0 disables it."},
        {"reset_offset", "(int) Phase of the reset. One-based update t resets when t % interval == offset % interval."},
        {"initial_v", "(int) Signed 24-bit initial voltage in Lava's shifted scale, restored on reset."},
        {"currents", "(list[int]) External activation per update, then zero, each in [-(2^18 + 32), 2^18 + 32]."},
    };
private:
    struct State
    {
        int32_t du{0};
        int32_t dv{0};
        int32_t vth{10};
        int32_t bias_mant{0};
        int32_t bias_exp{0};
        int32_t reset_interval{0};
        int32_t reset_offset{0};
        int32_t initial_v{0};
        int32_t u{0};
        int32_t v{0};
        std::vector<int32_t> currents;
        size_t cursor{0};
        size_t updates_seen{0};
    };
    std::vector<State> states;
};

// Sigma-delta neuron with the integer arithmetic of Lava's fixed-point sigma-delta models
// (lava/proc/sdn/models.py: PySigmaDeltaModelFixed, or PySigmaDeltaModelFixedCorrected when
// corrected is set), with the ReLU activation and without cumulative error. Each update
// returns the sent value as a graded spike when it is nonzero and sends nothing otherwise.
// Every state is signed 32 bits. Where Lava's int32 arithmetic would wrap, the update stops
// with std::overflow_error. It assumes the time-step buffer before the soma unit, which sums
// every input of a step.
class SigmaDeltaInt32Model : public SomaUnit
{
public:
    SigmaDeltaInt32Model() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override;
    // The states after the last update, the sent value, the input, and the largest magnitude
    // of each since the last reset.
    std::map<std::string, double> get_neuron_traces(size_t address) override;
    static inline const std::unordered_map<std::string, std::string> attributes{
        {"vth", "(int) Threshold in [0, 2^31 - 1], scaled as Lava's process scales it. A change is sent when its magnitude is >= vth. Required."},
        {"bias", "(int) Signed 32-bit bias added to sigma before the ReLU, scaled as vth is."},
        {"state_exp", "(int) Exponent in [0, 30]. The sent value is the change divided by 2^state_exp, rounded toward negative infinity."},
        {"corrected", "(bool) Use the rule of Lava's PySigmaDeltaModelFixedCorrected, which carries the remainder of the shift, instead of PySigmaDeltaModelFixed."},
        {"currents", "(list[int]) Signed 32-bit external input per update, added to the synaptic input, then zero."},
    };
private:
    struct State
    {
        int64_t vth{0};
        bool vth_set{false};
        int64_t bias{0};
        int64_t state_exp{0};
        bool corrected{false};
        std::vector<int32_t> currents;
        size_t cursor{0};
        int64_t sigma{0};
        int64_t act{0};
        int64_t residue{0};
        int64_t sent{0};
        int64_t input{0};
        int64_t max_abs_sigma{0};
        int64_t max_abs_act{0};
        int64_t max_abs_residue{0};
        int64_t max_abs_sent{0};
        int64_t max_abs_input{0};
    };
    std::vector<State> states;
};

// Sigma neuron with the integer arithmetic of Lava's PySigmaModelFixed
// (lava/proc/sdn/models.py): it adds its input to sigma and sends sigma at every update, as a
// graded spike when sigma is nonzero. Sigma is signed 32 bits and the update stops with
// std::overflow_error where Lava's int32 arithmetic would wrap. It has no attribute and
// assumes the time-step buffer before the soma unit.
class SigmaInt32Model : public SomaUnit
{
public:
    SigmaInt32Model() = default;
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t, const std::string &, const ModelAttribute &) override {}
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override;
    std::map<std::string, double> get_neuron_traces(size_t address) override;
private:
    struct State
    {
        int64_t sigma{0};
        int64_t sent{0};
        int64_t input{0};
        int64_t max_abs_sigma{0};
        int64_t max_abs_sent{0};
        int64_t max_abs_input{0};
    };
    std::vector<State> states;
};

// A numerical reference model, not a hardware-specific neuron ISA.
class Float32IfModel : public SomaUnit
{
public:
    Float32IfModel() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override { return states.at(address).voltage; }

    static inline const std::unordered_map<std::string, std::string> attributes{
        {"threshold", "(float) Positive firing threshold; comparison uses >=."},
        {"initial_voltage", "(float) Initial voltage, also restored on reset."},
        {"bias", "(float) Constant drive added on every update."},
        {"currents", "(list[float]) External current per update, then zero; excludes I/O cost."},
    };

private:
    struct State
    {
        float threshold{1.0F};
        float initial{0.0F};
        float voltage{0.0F};
        float bias{0.0F};
        std::vector<float> currents;
        size_t cursor{0};
    };
    std::vector<State> states;
};

class InputModel : public SomaUnit
{
public:
    InputModel() : gen(++instance_counter) { register_attributes(input_attributes); }
    InputModel(const InputModel &copy) = delete; // Because of random_device
    InputModel(InputModel &&other) = delete; // Because of random_device
    ~InputModel() override = default;
    InputModel &operator=(const InputModel &other) = delete;
    InputModel &operator=(InputModel &&other) = delete;

    void set_attribute_hw(const std::string &/*attribute_name*/, const ModelAttribute &/*param*/) override {};
    void set_attribute_neuron(size_t neuron_address, const std::string &attribute_name, const ModelAttribute &param) override;
    PipelineResult update(size_t neuron_address, std::optional<double> current_in, long int simulation_time) override;
    void reset() override { send_spike = false;
    }
    static inline const std::unordered_map<std::string, std::string> input_attributes{
        {"rate", "(float) Rate-based input encoding."},
        {"poisson", "(float) Randomized Poisson input encoding, i.e., random > poisson: spike, else no spike."},
        {"spikes", "(list[bool]) A per-time-step spike-train."},
    };

private:
    static inline unsigned int instance_counter = 0;
    std::vector<bool> spikes;
    size_t curr_spike{0UL};
    std::uniform_real_distribution<double> uniform_distribution{0.0, 1.0};
    // Intentionally use a fixed seed to get deterministic results across runs
    // To truly randomize, use:
    //std::random_device rd;
    //std::mt19937 gen{rd()};
    std::mt19937 gen;
    double poisson_probability{0.0};
    double rate{0.0};
    bool send_spike{false};
};

std::shared_ptr<PipelineUnit> model_get_pipeline_unit(const std::string &model_name);
NeuronResetModes model_parse_reset_mode(const std::string &str);

using ModelMap = std::map<std::string, const std::unordered_map<std::string, std::string>*>;
const ModelMap &get_builtin_models();
}


#endif
