#include "neural_network.h"
#include <gtest/gtest.h>

class NeuralNetworkTest : public testing::Test {
public:
    NeuralNetworkTest() {}

protected:
    NeuralNetwork neural_network_;
};

TEST_F(NeuralNetworkTest, ExampleTest) { 
    ASSERT_TRUE(true); 
}
