#include "matrix.h"
#include <gtest/gtest.h>

class MatrixTest : public testing::Test {
public:
    MatrixTest() {}

protected:
};

TEST_F(MatrixTest, MainConstructorTest) { 
    size_t rows = 5;
    size_t cols = 5;
    Matrix matrix(rows, cols);
    ASSERT_EQ(matrix.get_rows(), rows);
    ASSERT_EQ(matrix.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            ASSERT_NEAR(matrix[row][col], 0.0, 1e-9);
        }
    }
}

TEST_F(MatrixTest, CopyConstructorTest) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix matrix_1(rows, cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            matrix_1[row][col] = 666.0;
        }
    }
    Matrix matrix_2(matrix_1);
    ASSERT_EQ(matrix_1.get_rows(), matrix_2.get_rows());
    ASSERT_EQ(matrix_1.get_cols(), matrix_2.get_cols());
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            ASSERT_NEAR(matrix_2[row][col], 666.0, 1e-9);
        }
    }
}

TEST_F(MatrixTest, MoveConstructorTest) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix matrix_1(rows, cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            matrix_1[row][col] = 666.0;
        }
    }
    Matrix matrix_2(std::move(matrix_1));
    ASSERT_EQ(matrix_2.get_rows(), rows);
    ASSERT_EQ(matrix_2.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            ASSERT_NEAR(matrix_2[row][col], 666.0, 1e-9);
        }
    }
    ASSERT_EQ(matrix_1.get_rows(), 0);
    ASSERT_EQ(matrix_1.get_cols(), 0);
    ASSERT_EQ(matrix_1.is_empty(), true);
}
