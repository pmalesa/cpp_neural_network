#include "neural_network.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "loss.h"
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
    vector<ActivationFunction> activation_functions_1 = {ActivationFunction::ReLU, ActivationFunction::Softmax};
    NeuralNetwork nn_1(shape_1, activation_functions_1);

    EXPECT_TRUE(nn_1.get_shape().size() == 3);
    EXPECT_TRUE(nn_1.get_shape() == vector<size_t>({10, 20, 30}));
    EXPECT_TRUE(nn_1.get_shape()[0] == 10);
    EXPECT_TRUE(nn_1.get_shape()[1] == 20);
    EXPECT_TRUE(nn_1.get_shape()[2] == 30);
    EXPECT_TRUE(nn_1.get_activation_functions()[0] == ActivationFunction::ReLU);
    EXPECT_TRUE(nn_1.get_activation_functions()[1] == ActivationFunction::Softmax);
    EXPECT_TRUE(nn_1.get_weights().empty());
    EXPECT_FALSE(nn_1.is_built());

    vector<size_t> shape_2 = {11, 22, 33};
    vector<ActivationFunction> activation_functions_2 = {ActivationFunction::ReLU};

    EXPECT_THROW(NeuralNetwork nn_2(shape_2, activation_functions_2), std::logic_error);
}

TEST_F(NeuralNetworkTest, EraseMethodTest) { 
    vector<size_t> shape = {10, 20, 30};
    vector<ActivationFunction> activation_functions = {ActivationFunction::ReLU, ActivationFunction::Softmax};
    NeuralNetwork nn(shape, activation_functions);
    nn.erase();
    EXPECT_TRUE(nn.get_shape().empty());
    EXPECT_TRUE(nn.get_activation_functions().empty());
    EXPECT_TRUE(nn.get_weights().empty());
    EXPECT_FALSE(nn.is_built());    
}

TEST_F(NeuralNetworkTest, AddLayerMethodTest) {
    NeuralNetwork nn;

    EXPECT_THROW(nn.add_layer(10, ActivationFunction::ReLU), std::logic_error);
    EXPECT_THROW(nn.add_input_layer(0), std::logic_error);

    nn.add_input_layer(10);

    EXPECT_THROW(nn.add_input_layer(10), std::logic_error);

    nn.add_layer(20, ActivationFunction::ReLU);
    nn.add_layer(30, ActivationFunction::Softmax);

    EXPECT_TRUE(nn.get_shape() == vector<size_t>({10, 20, 30}));
    EXPECT_TRUE(nn.get_activation_functions().size() == 2);
    EXPECT_TRUE(nn.get_activation_functions()[0] == ActivationFunction::ReLU);
    EXPECT_TRUE(nn.get_activation_functions()[1] == ActivationFunction::Softmax);
}

TEST_F(NeuralNetworkTest, BuildMethodTest) {
    NeuralNetwork nn;

    nn.add_input_layer(10);

    // Try building with only input layer
    EXPECT_THROW(nn.build(), std::logic_error);
 
    nn.add_layer(3, ActivationFunction::Softmax);
    EXPECT_NO_THROW(nn.build());

    EXPECT_TRUE(nn.get_weights().size() == 1);
    EXPECT_THROW(nn.build(), std::logic_error); // Already built
    EXPECT_THROW(nn.add_layer(2, ActivationFunction::ReLU), std::logic_error);
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputShapeCheckSingleTest) {
    NeuralNetwork nn;
    nn.add_input_layer(3);
    nn.add_layer(2, ActivationFunction::Sigmoid);
    nn.build();

    Matrix input(3, 1);
    input.fill(0.5);

    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 2);
    EXPECT_TRUE(output.get_cols() == 1);
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputShapeCheckBatchTest) {
    NeuralNetwork nn;
    nn.add_input_layer(100);
    nn.add_layer(500);
    nn.add_layer(1000);
    nn.add_layer(500);
    nn.add_layer(4);
    nn.build();

    Matrix input(100, 15);
    input.fill(0.5);

    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 4);
    EXPECT_TRUE(output.get_cols() == 15);
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputValuesCheckSingleTest) {
    NeuralNetwork nn;
    nn.add_input_layer(4);
    nn.add_layer(8);
    nn.add_layer(16);
    nn.add_layer(2);
    nn.build();

    nn.get_weights()[0].fill(2.0);
    nn.get_weights()[1].fill(4.0);
    nn.get_weights()[2].fill(8.0);

    Matrix input(4, 1);
    input.fill(0.5);
    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 2);
    EXPECT_TRUE(output.get_cols() == 1);
    EXPECT_TRUE((output == il{ {25096.0}, {25096.0} }));
}

TEST_F(NeuralNetworkTest, ForwardMethodOutputValuesCheckBatchTest) {
    NeuralNetwork nn;
    nn.add_input_layer(4);
    nn.add_layer(8, ActivationFunction::Sigmoid);
    nn.add_layer(16, ActivationFunction::Sigmoid);
    nn.add_layer(2, ActivationFunction::Sigmoid);
    nn.build();

    nn.get_weights()[0].fill(2.0);
    nn.get_weights()[1].fill(4.0);
    nn.get_weights()[2].fill(8.0);

    Matrix input(4, 5);
    input.fill(0.5);
    Matrix output = nn.forward(input);
    EXPECT_TRUE(output.get_rows() == 2);
    EXPECT_TRUE(output.get_cols() == 5);
    EXPECT_TRUE((output == il{ {1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0} }));
}

TEST_F(NeuralNetworkTest, BackwardMethodTest) {
    NeuralNetwork nn;

    EXPECT_THROW(nn.backward({}, {}, 1e-5, LossFunction::BinaryCrossEntropy), std::logic_error);

    nn.add_input_layer(2);
    nn.add_layer(1, ActivationFunction::Sigmoid);
    nn.build();

    nn.get_weights()[0] = {
        {0.1},
        {0.2},
        {-0.1}
    };

    Matrix input = {
        {1.0},
        {2.0}
    };

    Matrix target = {
        {1.0}
    };

    constexpr double learning_rate = 0.1;
    constexpr double tolerance = 1e-9;

    nn.forward(input, true);
    nn.backward(input, target, learning_rate, LossFunction::MSE);

    
    const Matrix& updated_weights = nn.get_weights()[0];
    
    EXPECT_NEAR(updated_weights[0][0], 0.123691761847142, tolerance);
    EXPECT_NEAR(updated_weights[1][0], 0.223691761847142, tolerance);
    EXPECT_NEAR(updated_weights[2][0], -0.052616476305715, tolerance);
    EXPECT_THROW(nn.backward(input, target, learning_rate, LossFunction::MSE), std::logic_error);
}

TEST_F(NeuralNetworkTest, BackwardReducesCategoricalCrossEntropy) {
    NeuralNetwork nn;

    // 2 inputs -> 2 hidden neurons -> 2 output classes
    nn.add_input_layer(2);
    nn.add_layer(2, ActivationFunction::Sigmoid);
    nn.add_layer(2, ActivationFunction::Softmax);
    nn.build();

    nn.get_weights()[0] = {
        { 0.1, -0.2},  // biases
        { 0.3,  0.4},
        {-0.1,  0.2}
    };

    nn.get_weights()[1] = {
        { 0.05, -0.05},  // biases
        { 0.20, -0.10},
        {-0.30,  0.25}
    };

    // Two examples stored as columns
    Matrix input = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    Matrix target = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    // Before backprop
    Matrix prediction_before = nn.forward(input, true);
    double loss_before = Loss::categorical_cross_entropy(target, prediction_before);

    nn.backward(input, target, 0.1, LossFunction::CategoricalCrossEntropy);

    // After backprop
    Matrix prediction_after = nn.forward(input);
    double loss_after = Loss::categorical_cross_entropy(target, prediction_after);

    EXPECT_TRUE(loss_after < loss_before); // or EXPECT_LT(loss_after, loss_before)
}