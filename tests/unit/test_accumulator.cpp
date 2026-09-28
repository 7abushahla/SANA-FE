#include <gtest/gtest.h>

#include "attribute.hpp"
#include "models.hpp"

namespace
{
class TestAccumulatorModel : public ::testing::Test
{
protected:
    sanafe::AccumulatorModel model;

    sanafe::ModelAttribute make_attr_double(double val)
    {
        sanafe::ModelAttribute attr;
        attr.value = val;
        return attr;
    }
};
}

TEST_F(TestAccumulatorModel, IntegratesCurrent)
{
    model.update(0UL, 5.0, std::nullopt, 1L);
    auto result = model.update(0UL, std::nullopt, std::nullopt, 1L);

    ASSERT_TRUE(result.current.has_value());
    EXPECT_DOUBLE_EQ(result.current.value(), 5.0);
}

TEST_F(TestAccumulatorModel, AccumulatesChargeOverTime)
{
    model.update(0UL, 2.0, std::nullopt, 1L);
    model.update(0UL, 3.0, std::nullopt, 1L);
    auto result = model.update(0UL, std::nullopt, std::nullopt, 1L);

    ASSERT_TRUE(result.current.has_value());
    EXPECT_DOUBLE_EQ(result.current.value(), 5.0);
}

TEST_F(TestAccumulatorModel, UnknownAttributeDoesNotThrow)
{
    sanafe::ModelAttribute attr;
    attr.value = 42.0;
    EXPECT_NO_THROW(model.set_attribute_neuron(0, "unknown_attribute", attr));
}

TEST_F(TestAccumulatorModel, ResizesBeyond1024AndKeepsAddressesIndependent)
{
    model.update(4096, 7.0, std::nullopt, 1);
    model.update(0, -3.0, std::nullopt, 1);
    EXPECT_EQ(model.update(4096, std::nullopt, std::nullopt, 1).current, 7.0);
    EXPECT_EQ(model.update(0, std::nullopt, std::nullopt, 1).current, -3.0);
}

TEST_F(TestAccumulatorModel, ResetRestartsTimestamps)
{
    model.update(0, 9.0, std::nullopt, 20);
    model.reset();
    model.update(0, 3.0, std::nullopt, 1);
    EXPECT_EQ(model.update(0, 4.0, std::nullopt, 2).current, 4.0);
}

TEST(DelayedAccumulator, ResizesAndResetClearsQueueAndTimestamps)
{
    sanafe::AccumulatorWithDelayModel model;
    sanafe::ModelAttribute delay;
    delay.value = 2;
    model.set_attribute_edge(0, "delay", delay);
    model.update(4096, 7.0, 0, 1);
    EXPECT_FALSE(model.update(4096, std::nullopt, 0, 3).current.has_value());
    EXPECT_EQ(model.update(4096, std::nullopt, 0, 4).current, 7.0);
    model.update(4096, 99.0, 0, 20);
    model.reset();
    model.update(4096, -3.0, 0, 1);
    EXPECT_FALSE(model.update(4096, std::nullopt, 0, 3).current.has_value());
    EXPECT_EQ(model.update(4096, std::nullopt, 0, 4).current, -3.0);
    EXPECT_FALSE(model.update(4096, std::nullopt, 0, 5).current.has_value());
}

TEST(IntegerAccumulator, ChecksEveryAdditionAndPreservesSumOnOverflow)
{
    sanafe::IntegerAccumulatorModel model;
    model.update(2048, 2147483647.0, std::nullopt, 1);
    EXPECT_THROW(model.update(2048, 1.0, std::nullopt, 1), std::overflow_error);
    EXPECT_EQ(model.update(2048, -1.0, std::nullopt, 1).current, 2147483646.0);
    model.update(0, -2147483648.0, std::nullopt, 1);
    EXPECT_THROW(model.update(0, -1.0, std::nullopt, 1), std::overflow_error);
    EXPECT_EQ(model.update(0, 1.0, std::nullopt, 1).current, -2147483647.0);
    EXPECT_THROW(model.update(0, .5, std::nullopt, 1), std::invalid_argument);
    EXPECT_THROW(model.update(0, 2147483648.0, std::nullopt, 1), std::invalid_argument);
    model.reset();
    model.update(2048, 9.0, std::nullopt, 1);
    EXPECT_EQ(model.update(2048, 4.0, std::nullopt, 2).current, 4.0);
}

TEST(IntegerAccumulator, ResetClearsTimestampsAfterLongRun)
{
    sanafe::IntegerAccumulatorModel model;
    model.update(2048, 100.0, std::nullopt, 20);
    model.reset();
    model.update(2048, 3.0, std::nullopt, 1);
    EXPECT_EQ(model.update(2048, 4.0, std::nullopt, 2).current, 4.0);
}
