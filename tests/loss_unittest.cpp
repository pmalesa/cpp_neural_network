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
    EXPECT_TRUE(true);
}

TEST_F(LossTest, MAETest) {
    EXPECT_TRUE(true);
}