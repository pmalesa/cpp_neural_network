#include "model.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "loss.h"
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
    Model model;
    model.add_input_layer(2);
    model.add_layer(3, ActivationFunction::ReLU);
    model.add_layer(2, ActivationFunction::Softmax);

    EXPECT_TRUE(model.shape() == vector<size_t>({2, 3, 2}));
}

TEST_F(ModelTest, FitRejectsInvalidArguments) {
    // ...
}

TEST_F(ModelTest, FitReducesLossTest) {
    Model model;
    model.add_input_layer(2);
    model.add_layer(2, ActivationFunction::Softmax);

    Matrix X = {
        {-1.0, -1.0, 1.0, 1.0},
        {-1.0,  1.0, -1.0, 1.0}
    };

    Matrix y = {
        {1.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 1.0}
    };

    model.fit(X, y, 1, 0.01, LossFunction::CategoricalCrossEntropy);

    Matrix pred_before = model.predict(X);
    double loss_before = Loss::categorical_cross_entropy(y, pred_before);

    model.fit(X, y, 1, 0.01, LossFunction::CategoricalCrossEntropy);

    Matrix pred_after = model.predict(X);
    double loss_after = Loss::categorical_cross_entropy(y, pred_after);

    EXPECT_TRUE(loss_after < loss_before);
}

TEST_F(ModelTest, ClearMethodTest) {
    Model model;
    model.add_input_layer(2);
    model.add_layer(3, ActivationFunction::ReLU);
    model.add_layer(2, ActivationFunction::Softmax);
    model.clear();

    EXPECT_TRUE(model.shape().size() == 0);
}