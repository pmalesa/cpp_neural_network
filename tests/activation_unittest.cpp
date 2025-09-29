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