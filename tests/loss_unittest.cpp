#include "matrix.h"
#include "loss.h"
#include <gtest/gtest.h>

using il = std::initializer_list<std::initializer_list<double>>;

class LossTest : public testing::Test {
public:
    LossTest() {}

protected:
};

TEST_F(LossTest, MSETest) {
    Matrix y_true = { {1.0}, {2.0}, {3.0} };
    Matrix y_pred = { {2.0}, {2.0}, {4.0} };
    EXPECT_TRUE(Loss::mse(y_true, y_pred) == 2.0 / 3.0);
}

TEST_F(LossTest, MSEDerivativeTest) {
    Matrix y_true = { {1.0}, {2.0}, {3.0} };
    Matrix y_pred = { {2.0}, {2.0}, {4.0} };
    Matrix grad = Loss::mse_derivative(y_true, y_pred);
    EXPECT_TRUE(grad.get_rows() == 3);
    EXPECT_TRUE(grad.get_cols() == 1);
    EXPECT_TRUE((grad == il{ {2.0 / 3.0}, {0.0}, {2.0 / 3.0} }));
}