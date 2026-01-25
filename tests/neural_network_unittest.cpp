#include "model.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "spdlog/spdlog.h"
#include <stdexcept>

using il = std::initializer_list<std::initializer_list<double>>;

class NeuralNetworkTest : public testing::Test {
public:
    NeuralNetworkTest() {
        spdlog::set_level(spdlog::level::off);
    }

    ~NeuralNetworkTest() {
        spdlog::set_level(spdlog::level::info);
    }

protected:
};

TEST_F(NeuralNetworkTest, ConstructorTest) {
    vector<size_t> shape_1 = {10, 20, 30};
    vector<ActivationFunction> activation_functions_1 = {ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::Softmax};
    NeuralNetwork nn_1(shape_1, activation_functions_1);
    EXPECT_TRUE(nn_1.get_shape().size() == 3);
    EXPECT_TRUE(nn_1.get_shape()[0] == 10);
    EXPECT_TRUE(nn_1.get_shape()[1] == 20);
    EXPECT_TRUE(nn_1.get_shape()[2] == 30);
    EXPECT_TRUE(nn_1.get_activation_functions()[0] == ActivationFunction::ReLU);
    EXPECT_TRUE(nn_1.get_activation_functions()[1] == ActivationFunction::ReLU);
    EXPECT_TRUE(nn_1.get_activation_functions()[2] == ActivationFunction::Softmax);
    EXPECT_TRUE(nn_1.get_weights().empty());
    EXPECT_TRUE(not nn_1.is_built());
    vector<size_t> shape_2 = {10, 20, 30};
    vector<ActivationFunction> activation_functions_2 = {ActivationFunction::ReLU, ActivationFunction::Softmax};
    EXPECT_THROW(NeuralNetwork nn_2(shape_2, activation_functions_2), std::logic_error);
}

TEST_F(NeuralNetworkTest, EraseMethodTest) { 
    vector<size_t> shape = {10, 20, 30};
    vector<ActivationFunction> activation_functions = {ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::Softmax};
    NeuralNetwork nn(shape, activation_functions);
    nn.erase();
    EXPECT_TRUE(nn.get_shape().empty());
    EXPECT_TRUE(nn.get_activation_functions().empty());
    EXPECT_TRUE(nn.get_weights().empty());
    EXPECT_TRUE(not nn.is_built());    
}

TEST_F(NeuralNetworkTest, BuildMethodTest) {
    NeuralNetwork nn;

    // Try building with 0 layers
    EXPECT_THROW(nn.build(), std::logic_error);

    // Test building with 1 layer
    nn.add_layer(10);
    EXPECT_THROW(nn.build(), std::logic_error);

    // Add second layer
    nn.add_layer(20);

    // Add third layer
    nn.add_layer(30);

    // Add output layer
    nn.add_layer(1, LayerType::Output);

    nn.build();
    EXPECT_TRUE(nn.is_built());
    for (size_t layer = 0; layer < nn.get_shape().size() - 1; ++layer) {
        Matrix& weights = nn.get_weights()[layer];
        EXPECT_TRUE(weights.get_rows() == nn.get_shape()[layer] + 1);
        EXPECT_TRUE(weights.get_cols() == nn.get_shape()[layer + 1]);
    }
    EXPECT_TRUE(nn.get_weights().size() == nn.get_activation_functions().size());
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputShapeCheckSingleTest) {
    NeuralNetwork nn;

    // Add first layer
    nn.add_layer(3, ActivationFunction::Sigmoid);

    // Add output layer
    nn.add_layer(2, LayerType::Output);

    nn.build();

    Matrix input(3, 1);
    input.fill(0.5);

    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 2);
    EXPECT_TRUE(output.get_cols() == 1);
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputShapeCheckBatchTest) {
    NeuralNetwork nn;

    // Add first layer
    nn.add_layer(100);

    // Add second layer
    nn.add_layer(500);

    // Add third layer
    nn.add_layer(1000);

    // Add fourth layer
    nn.add_layer(500);

    // Add output layer
    nn.add_layer(4, LayerType::Output);

    nn.build();

    Matrix input(100, 15);
    input.fill(0.5);

    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 4);
    EXPECT_TRUE(output.get_cols() == 15);
}
/* 
TEST_F(NeuralNetworkTest, ForwardMethodOutputValuesCheckSingleTest) {
    NeuralNetwork nn;

    // Add first layer
    nn.add_layer(4);

    // Add second layer
    nn.add_layer(8);

    // Add third layer
    nn.add_layer(16);

    // Add output layer
    nn.add_layer(2, LayerType::Output);

    nn.build();

    nn.weights[0].fill(2.0);
    nn.weights[1].fill(4.0);
    nn.weights[2].fill(8.0);

    Matrix input(4, 1);
    input.fill(0.5);
    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 2);
    EXPECT_TRUE(output.get_cols() == 1);
    EXPECT_TRUE((output == il{ {25096.0}, {25096.0} }));
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputValuesCheckBatchTest) {
    NeuralNetwork nn;
    // Add first layer
    ++nn.n_layers;
    nn.shape.push_back(4);
    nn.activation_functions.push_back(ActivationFunction::Sigmoid);

    // Add second layer
    ++nn.n_layers;
    nn.shape.push_back(8);
    nn.activation_functions.push_back(ActivationFunction::Sigmoid);

    // Add third layer
    ++nn.n_layers;
    nn.shape.push_back(16);
    nn.activation_functions.push_back(ActivationFunction::Sigmoid);

    // Add output layer
    ++nn.n_layers;
    nn.shape.push_back(2);

    nn.build();

    nn.weights[0].fill(2.0);
    nn.weights[1].fill(4.0);
    nn.weights[2].fill(8.0);

    Matrix input(4, 5);
    input.fill(0.5);
    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 2);
    EXPECT_TRUE(output.get_cols() == 5);
    EXPECT_TRUE((output == il{ {1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0} }));
}
 */
TEST_F(NeuralNetworkTest, BackwardMethodTest) {
    NeuralNetwork nn;
    EXPECT_THROW(nn.backward({}, {}, 1e-5), std::logic_error);
    // ...
}
