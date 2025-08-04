#include "matrix.h"
#include <gtest/gtest.h>

class MatrixTest : public testing::Test {
public:
    MatrixTest() {}

protected:
};

TEST_F(MatrixTest, MainConstructorTest) { 
    size_t rows = 4;
    size_t cols = 5;
    Matrix matrix(rows, cols);
    ASSERT_EQ(matrix.get_rows(), rows);
    ASSERT_EQ(matrix.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            ASSERT_NEAR(matrix.at(row, col), 0.0, 1e-9);
        }
    }
}
