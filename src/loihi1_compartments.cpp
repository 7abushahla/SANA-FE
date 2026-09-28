#include "loihi1_compartments.hpp"

#include <cmath>
#include <stdexcept>

namespace
{
constexpr int64_t signed24_min = -(1LL << 23);
constexpr int64_t signed24_max = (1LL << 23) - 1;

int64_t checked_integer(const double value, const int64_t minimum,
        const int64_t maximum, const std::string &name)
{
    if (!std::isfinite(value) || std::trunc(value) != value ||
            value < static_cast<double>(minimum) ||
            value > static_cast<double>(maximum))
    {
        throw std::invalid_argument("paired_compartment_if " + name +
                " must be an integer in [" + std::to_string(minimum) +
                ", " + std::to_string(maximum) + "]");
    }
    return static_cast<int64_t>(value);
}
}

void sanafe::PairedCompartmentIfModel::set_attribute_neuron(
        const size_t address, const std::string &name, const ModelAttribute &param)
{
    if (states.size() <= address)
    {
        states.resize(address + 1);
    }
    State &state = states.at(address);
    configuration_validated = false;
    if (name == "pair_role")
    {
        const std::string role = static_cast<std::string>(param);
        if (role == "dendrite") { state.role = Role::dendrite; }
        else if (role == "soma") { state.role = Role::soma; }
        else if (role == "hard") { state.role = Role::hard; }
        else if (role == "dummy") { state.role = Role::dummy; }
        else { throw std::invalid_argument("Invalid paired_compartment_if pair_role"); }
    }
    else if (name == "threshold")
    {
        state.threshold = checked_integer(static_cast<double>(param),
                1, signed24_max, name);
    }
    else if (name == "bias")
    {
        state.bias = checked_integer(static_cast<double>(param),
                signed24_min, signed24_max, name);
    }
    else if (name == "initial_voltage")
    {
        checked_integer(static_cast<double>(param), 0, 0, name);
    }
    else if (framework_attributes.find(name) == framework_attributes.end())
    {
        throw std::invalid_argument("Unsupported paired_compartment_if attribute: " + name);
    }
}

void sanafe::PairedCompartmentIfModel::validate_configuration()
{
    states.resize(neuron_count);
    for (size_t address = 0; address < states.size(); ++address)
    {
        const State &state = states.at(address);
        if (state.role == Role::unset)
        {
            throw std::invalid_argument("paired_compartment_if requires pair_role");
        }
        if (state.role == Role::dendrite &&
                (address + 1 >= states.size() || states.at(address + 1).role != Role::soma))
        {
            throw std::invalid_argument("Dendrite must immediately precede its soma");
        }
        if (state.role == Role::soma &&
                (address == 0 || states.at(address - 1).role != Role::dendrite))
        {
            throw std::invalid_argument("Soma requires an immediately preceding dendrite");
        }
        if ((state.role == Role::soma || state.role == Role::hard) && state.threshold == 0)
        {
            throw std::invalid_argument("Spiking compartment requires threshold");
        }
        if ((state.role == Role::soma || state.role == Role::dummy) && state.bias != 0)
        {
            throw std::invalid_argument("Soma/dummy bias must be zero");
        }
    }
    configuration_validated = true;
}

sanafe::PipelineResult sanafe::PairedCompartmentIfModel::update(
        const size_t address, const std::optional<double> current,
        const long int /*timestep*/)
{
    if (!configuration_validated)
    {
        validate_configuration();
    }
    if (address != next_address || address >= states.size())
    {
        throw std::runtime_error("Compartments must update once in ascending mapped order");
    }
    State &state = states.at(address);
    const int64_t incoming = checked_integer(current.value_or(0.0),
            signed24_min, signed24_max, "input current");
    bool spike = false;
    if (state.role == Role::soma || state.role == Role::dummy)
    {
        if (incoming != 0)
        {
            throw std::invalid_argument("Soma/dummy synaptic input must be zero");
        }
        // Public NxTF joinOp=6 topology interpreted as pass-stack voltage.
        // No general stack machine or BAP microcode semantics are asserted.
        state.voltage = state.role == Role::soma ? states.at(address - 1).voltage : 0;
    }
    else
    {
        const int64_t integrated = state.voltage + incoming + state.bias;
        state.voltage = checked_integer(static_cast<double>(integrated),
                signed24_min, signed24_max, "accumulated voltage");
    }
    state.current = incoming;
    if (state.role == Role::soma || state.role == Role::hard)
    {
        spike = state.voltage > state.threshold;
        if (spike)
        {
            state.voltage = 0;
        }
    }
    // The dendrite is never reset here. An explicit recurrent graph edge
    // delivers inhibition during the next update, including its event cost.
    next_address = (address + 1) % states.size();
    PipelineResult result;
    result.status = state.role == Role::dummy ? idle : (spike ? fired : updated);
    return result;
}

void sanafe::PairedCompartmentIfModel::reset()
{
    for (State &state : states)
    {
        state.voltage = 0;
        state.current = 0;
    }
    next_address = 0;
}

double sanafe::PairedCompartmentIfModel::get_potential(const size_t address)
{
    return static_cast<double>(states.at(address).voltage);
}

std::map<std::string, double> sanafe::PairedCompartmentIfModel::get_neuron_traces(
        const size_t address)
{
    const State &state = states.at(address);
    return {{"u", static_cast<double>(state.current)},
            {"v", static_cast<double>(state.voltage)}};
}
