#include "neural_network.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "logger.h"
#include <stdexcept>

static Logger& logger = Logger::instance();

class NeuralNetworkTest : public testing::Test {
public:
    NeuralNetworkTest() {
        logger.set_printing_enabled(false);
    }

    ~NeuralNetworkTest() {
        logger.set_printing_enabled(true);
    }

protected:
    NeuralNetwork neural_network_;
};

TEST_F(NeuralNetworkTest, ConstructorTest) {
    vector<size_t> network_shape_1 = {10, 20, 30};
    vector<ActivationFunction> activation_functions_1 = {ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::Softmax};
    NeuralNetwork nn_1(network_shape_1, activation_functions_1);
    EXPECT_TRUE(nn_1.get_shape().size() == 3);
    EXPECT_TRUE(nn_1.get_shape()[0] == 10);
    EXPECT_TRUE(nn_1.get_shape()[1] == 20);
    EXPECT_TRUE(nn_1.get_shape()[2] == 30);
    EXPECT_TRUE(nn_1.get_activation_functions()[0] == ActivationFunction::ReLU);
    EXPECT_TRUE(nn_1.get_activation_functions()[1] == ActivationFunction::ReLU);
    EXPECT_TRUE(nn_1.get_activation_functions()[2] == ActivationFunction::Softmax);
    EXPECT_TRUE(nn_1.get_weights().empty());
    EXPECT_TRUE(not nn_1.is_built());
    vector<size_t> network_shape_2 = {10, 20, 30};
    vector<ActivationFunction> activation_functions_2 = {ActivationFunction::ReLU, ActivationFunction::Softmax};
    EXPECT_THROW(NeuralNetwork nn_2(network_shape_2, activation_functions_2), std::logic_error);
}

TEST_F(NeuralNetworkTest, InitMethodTest) { 
    NeuralNetwork nn;
    nn.init(5);
    EXPECT_TRUE(nn.get_shape().size() == 1);
    EXPECT_TRUE(nn.get_activation_functions().size() == 1);
    EXPECT_TRUE(nn.get_weights().empty());
    EXPECT_TRUE(not nn.is_built());    
}

TEST_F(NeuralNetworkTest, EraseMethodTest) { 
    vector<size_t> network_shape = {10, 20, 30};
    vector<ActivationFunction> activation_functions = {ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::Softmax};
    NeuralNetwork nn(network_shape, activation_functions);
    nn.erase();
    EXPECT_TRUE(nn.get_shape().empty());
    EXPECT_TRUE(nn.get_activation_functions().empty());
    EXPECT_TRUE(nn.get_weights().empty());
    EXPECT_TRUE(not nn.is_built());    
}

TEST_F(NeuralNetworkTest, AddLayerMethodTest) {
    NeuralNetwork nn;
    EXPECT_THROW(nn.add_layer(666), std::logic_error);
    nn.init(10);
    nn.add_layer(20);
    nn.add_layer(30);
    nn.add_layer(1);    
    EXPECT_TRUE(nn.get_shape()[0] == 10);
    EXPECT_TRUE(nn.get_shape()[1] == 20);
    EXPECT_TRUE(nn.get_shape()[2] == 30);
    EXPECT_TRUE(nn.get_shape()[3] == 1);
    nn.build();
    EXPECT_THROW(nn.add_layer(666), std::logic_error);
}

TEST_F(NeuralNetworkTest, BuildMethodTest) {
    NeuralNetwork nn;
    EXPECT_THROW(nn.build(), std::logic_error);
    nn.init(10);
    EXPECT_THROW(nn.build(), std::logic_error);
    nn.add_layer(20);
    nn.add_layer(30);
    nn.add_layer(1); 
    nn.build();
    EXPECT_TRUE(nn.is_built());
    for (size_t layer = 0; layer < nn.get_shape().size() - 1; ++layer) {
        Matrix& weights = nn.get_weights()[layer];
        EXPECT_TRUE(weights.get_rows() == nn.get_shape()[layer] + 1);
        EXPECT_TRUE(weights.get_cols() == nn.get_shape()[layer + 1]);
    }
}
