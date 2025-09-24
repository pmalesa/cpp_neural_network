#include "neural_network.h"
#include <gtest/gtest.h>

class NeuralNetworkTest : public testing::Test {
public:
    NeuralNetworkTest() {}

protected:
    NeuralNetwork neural_network_;
};

TEST_F(NeuralNetworkTest, ConstructorTest) {
    NeuralNetwork nn({10, 20, 30});
    EXPECT_TRUE(nn.get_shape().size() == 3);
    EXPECT_TRUE(nn.get_shape()[0] == 10);
    EXPECT_TRUE(nn.get_shape()[1] == 20);
    EXPECT_TRUE(nn.get_shape()[2] == 30);
    EXPECT_TRUE(nn.get_activation_functions().empty());
    EXPECT_TRUE(nn.get_weights().empty());
    EXPECT_TRUE(not nn.is_classifier());
    EXPECT_TRUE(not nn.is_built());
}

TEST_F(NeuralNetworkTest, InitMethodTest) { 
    ASSERT_TRUE(true); 
}

TEST_F(NeuralNetworkTest, EraseMethodTest) { 
    ASSERT_TRUE(true); 
}
