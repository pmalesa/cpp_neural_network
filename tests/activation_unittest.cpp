#include "matrix.h"
#include "activation.h"
#include <gtest/gtest.h>

using il = std::initializer_list<std::initializer_list<double>>;

class ActivationTest : public testing::Test {
public:
    ActivationTest() {}

protected:
};

TEST_F(ActivationTest, ReluTest) {
    EXPECT_NEAR(Activation::relu(-3.0), 0.0, 1e-9);
    EXPECT_NEAR(Activation::relu(-2.0), 0.0, 1e-9);
    EXPECT_NEAR(Activation::relu(-1.0), 0.0, 1e-9);
    EXPECT_NEAR(Activation::relu(0.0), 0.0, 1e-9);
    EXPECT_NEAR(Activation::relu(1.0), 1.0, 1e-9);
    EXPECT_NEAR(Activation::relu(2.0), 2.0, 1e-9);
    EXPECT_NEAR(Activation::relu(3.0), 3.0, 1e-9);
    Matrix mat = { {-1.0, -2.0, -3.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 3.0} };
    EXPECT_TRUE((Activation::relu(mat) == il{ {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 3.0} }));
}

TEST_F(ActivationTest, SigmoidTest) {
    EXPECT_NEAR(Activation::sigmoid(-3.0), 0.04742587317751919, 1e-9);
    EXPECT_NEAR(Activation::sigmoid(-2.0), 0.11920292202204383, 1e-9);
    EXPECT_NEAR(Activation::sigmoid(-1.0), 0.26894142136992605, 1e-9);
    EXPECT_NEAR(Activation::sigmoid(0.0), 0.5, 1e-9);
    EXPECT_NEAR(Activation::sigmoid(1.0), 0.731058578630074, 1e-9);
    EXPECT_NEAR(Activation::sigmoid(2.0), 0.8807970779779563, 1e-9);
    EXPECT_NEAR(Activation::sigmoid(3.0), 0.9525741268224809, 1e-9);
    Matrix mat = { {-1.0, -2.0, -3.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 3.0} };
    EXPECT_TRUE((Activation::sigmoid(mat) == il{ 
        {0.04742587317751919, 0.11920292202204383, 0.26894142136992605},
        {0.5, 0.5, 0.5}, 
        {0.731058578630074, 0.8807970779779563, 0.9525741268224809} 
    }));
}

TEST_F(ActivationTest, TanhTest) {
    EXPECT_NEAR(Activation::tanh(-3.0), -0.99505475368, 1e-9);
    EXPECT_NEAR(Activation::tanh(-2.0), -0.96402758007, 1e-9);
    EXPECT_NEAR(Activation::tanh(-1.0), -0.76159415595, 1e-9);
    EXPECT_NEAR(Activation::tanh(0.0), 0.0, 1e-9);
    EXPECT_NEAR(Activation::tanh(1.0), 0.76159415595, 1e-9);
    EXPECT_NEAR(Activation::tanh(2.0), 0.96402758007 , 1e-9);
    EXPECT_NEAR(Activation::tanh(3.0), 0.99505475368, 1e-9);
    Matrix mat = { {-1.0, -2.0, -3.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 3.0} };
    EXPECT_TRUE((Activation::tanh(mat) == il{ 
        {-0.99505475368, -0.96402758007, -0.76159415595},
        {0.0, 0.0, 0.0}, 
        {0.76159415595, 0.96402758007, 0.99505475368} 
    }));
}
