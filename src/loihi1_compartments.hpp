// Restricted interpretation of a public NxTF compartment topology.
// This is a source-derived candidate, not a validated Loihi1 ISA model.
#ifndef SANAFE_LOIHI1_COMPARTMENTS_HPP
#define SANAFE_LOIHI1_COMPARTMENTS_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pipeline.hpp"

namespace sanafe
{
class PairedCompartmentIfModel : public SomaUnit
{
public:
    PairedCompartmentIfModel() { register_attributes(attributes); }
    void set_attribute_hw(const std::string &, const ModelAttribute &) override {}
    void set_attribute_neuron(size_t address, const std::string &name,
            const ModelAttribute &param) override;
    PipelineResult update(size_t address, std::optional<double> current,
            long int timestep) override;
    void reset() override;
    double get_potential(size_t address) override;
    std::map<std::string, double> get_neuron_traces(size_t address) override;

    static inline const std::unordered_map<std::string, std::string> attributes{
            {"pair_role", "Required role: dendrite, soma, hard, or dummy."},
            {"threshold", "Positive effective signed24 integer; strict > comparison."},
            {"bias", "Effective signed24 integer bias; zero for soma/dummy."},
            {"initial_voltage", "Only zero initial voltage is supported."}};

private:
    enum class Role { unset, dendrite, soma, hard, dummy };
    struct State
    {
        Role role{Role::unset};
        int64_t bias{0};
        int64_t threshold{0};
        int64_t voltage{0};
        int64_t current{0};
    };
    std::vector<State> states;
    size_t next_address{0};
    bool configuration_validated{false};
    void validate_configuration();
};
}
#endif
