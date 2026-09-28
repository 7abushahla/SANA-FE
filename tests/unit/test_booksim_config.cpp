#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include <booksim_lib.hpp>
#include <globals.hpp>
#include <ryml.hpp>

#include "arch.hpp"
#include "chip.hpp"
#include "network.hpp"
#include "yaml_arch.hpp"

TEST(BookSimConfigurationTest, ParsesOptionalNetworkSettings)
{
    std::string yaml = R"(attributes:
  width: 8
  height: 4
  link_buffer_size: 16
  booksim:
    subnets: 2
    packet_size: 3
    clock_period: 2.5e-9
    num_vcs: 2
    vc_buf_size: 12
    use_noc_latency: true
)";
    ryml::EventHandlerTree handler{};
    ryml::Parser parser(&handler, ryml::ParserOptions().locations(true));
    auto tree = ryml::parse_in_place(&parser, yaml.data());
    const auto parsed = sanafe::description_parse_noc_configuration_yaml(
            parser, tree.rootref()["attributes"]);

    EXPECT_EQ(parsed.booksim.subnets, 2);
    EXPECT_EQ(parsed.booksim.packet_size, 3);
    EXPECT_DOUBLE_EQ(parsed.booksim.clock_period, 2.5e-9);
    EXPECT_EQ(parsed.booksim.num_vcs, 2);
    EXPECT_EQ(parsed.booksim.vc_buf_size, 12);
    EXPECT_TRUE(parsed.booksim.use_noc_latency);
}

TEST(BookSimLifecycleTest, ReleasesTrafficManagerAfterCycleTimestep)
{
    const std::filesystem::path root(SANAFE_ROOT_PATH);
    auto arch = sanafe::load_arch(root / "sanafe/examples/example_chip.yaml");
    auto net = sanafe::load_net(root / "sanafe/examples/example_snn.yaml", arch);
    sanafe::SpikingChip chip(arch);
    chip.load(net);

    chip.sim(1, sanafe::timing_model_cycle_accurate, 0);
    const bool released = SimContext::get().trafficManager == nullptr;
    if (!released)
    {
        booksim_close(); // Keep the shared BookSim state clean after a red test.
    }
    EXPECT_TRUE(released);
}
