#include "model.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "spdlog/spdlog.h"
#include <stdexcept>

using il = std::initializer_list<std::initializer_list<double>>;

class ModelTest : public testing::Test {
public:
    ModelTest() {
        spdlog::set_level(spdlog::level::off);
    }

    ~ModelTest() {
        spdlog::set_level(spdlog::level::info);
    }

protected:
};

TEST_F(ModelTest, ConstructorTest) {
    vector<size_t> shape = {10, 20, 30};
    vector<ActivationFunction> activation_functions = {ActivationFunction::ReLU, ActivationFunction::Softmax};
    EXPECT_THROW(Model model(shape, activation_functions), std::logic_error);
}

TEST_F(ModelTest, AddLayerMethodTest) {
    Model model;
    model.add_layer(10);
    model.add_layer(20);
    model.add_layer(30);
    model.add_layer(1, LayerType::Output);    
    EXPECT_TRUE(model.shape()[0] == 10);
    EXPECT_TRUE(model.shape()[1] == 20);
    EXPECT_TRUE(model.shape()[2] == 30);
    EXPECT_TRUE(model.shape()[3] == 1);
    model.fit({}, {}, 100, 1e-5, LossFunction::MAE);
    EXPECT_THROW(model.add_layer(666), std::logic_error);
}
