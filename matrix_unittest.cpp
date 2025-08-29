#include "matrix.h"
#include <gtest/gtest.h>
#include <vector>

using std::vector;

class MatrixTest : public testing::Test {
public:
    MatrixTest() {}

protected:
};

TEST_F(MatrixTest, EqualsOperatorMat) {
    Matrix matrix_1(5, 5, 6.66);
    Matrix matrix_2(5, 5, 6.66);
    Matrix matrix_3(4, 4, 6.66);
    Matrix matrix_4(5, 5, 1.00);
    EXPECT_EQ(matrix_1 == matrix_2, true);
    EXPECT_EQ(matrix_2 == matrix_3, false);
    EXPECT_EQ(matrix_3 == matrix_4, false);
    EXPECT_EQ(matrix_4 == matrix_1, false);
}

TEST_F(MatrixTest, EqualsOperatorVec) {
    Matrix matrix(3, 3, 5.0);
    vector<vector<double>> data_1 = { {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} };
    vector<vector<double>> data_2 = { {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0} };
    vector<vector<double>> data_3 = { {5.0, 5.0}, {5.0, 5.0} };
    vector<vector<double>> data_4 = {};
    EXPECT_EQ(matrix == data_1, true);
    EXPECT_EQ(matrix == data_2, false);
    EXPECT_EQ(matrix == data_3, false);
    EXPECT_EQ(matrix == data_4, false);
}

TEST_F(MatrixTest, NotEqualsOperatorMat) {
    Matrix matrix_1(5, 5, 6.66);
    Matrix matrix_2(5, 5, 6.66);
    Matrix matrix_3(4, 4, 6.66);
    Matrix matrix_4(5, 5, 1.00);
    EXPECT_EQ(matrix_1 != matrix_2, false);
    EXPECT_EQ(matrix_2 != matrix_3, true);
    EXPECT_EQ(matrix_3 != matrix_4, true);
    EXPECT_EQ(matrix_4 != matrix_1, true);
}

TEST_F(MatrixTest, NotEqualsOperatorVec) {
    Matrix matrix(3, 3, 5.0);
    vector<vector<double>> data_1 = { {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} };
    vector<vector<double>> data_2 = { {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0} };
    vector<vector<double>> data_3 = { {5.0, 5.0}, {5.0, 5.0} };
    vector<vector<double>> data_4 = {};
    EXPECT_EQ(matrix != data_1, false);
    EXPECT_EQ(matrix != data_2, true);
    EXPECT_EQ(matrix != data_3, true);
    EXPECT_EQ(matrix != data_4, true);
}

TEST_F(MatrixTest, MainConstructorTest) { 
    size_t rows = 5;
    size_t cols = 5;
    Matrix matrix(rows, cols);
    ASSERT_EQ(matrix.get_rows(), rows);
    ASSERT_EQ(matrix.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            EXPECT_NEAR(matrix[row][col], 0.0, 1e-9);
        }
    }
    EXPECT_THROW(Matrix(0, 5), std::invalid_argument);
    EXPECT_THROW(Matrix(5, 0), std::invalid_argument);
}

TEST_F(MatrixTest, MainConstructorValTest) { 
    size_t rows = 5;
    size_t cols = 5;
    double val = 666.0;
    Matrix matrix(rows, cols, val);
    ASSERT_EQ(matrix.get_rows(), rows);
    ASSERT_EQ(matrix.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            EXPECT_NEAR(matrix[row][col], val, 1e-9);
        }
    }
    EXPECT_THROW(Matrix(0, 5, 1.0), std::invalid_argument);
    EXPECT_THROW(Matrix(5, 0, 1.0), std::invalid_argument);
}

TEST_F(MatrixTest, CopyConstructorTest) {
    size_t rows = 5;
    size_t cols = 5;
    double val = 666.0;
    Matrix matrix_1(rows, cols, val);
    Matrix matrix_2(matrix_1);
    EXPECT_EQ(matrix_1.get_rows(), matrix_2.get_rows());
    EXPECT_EQ(matrix_1.get_cols(), matrix_2.get_cols());
    EXPECT_EQ(matrix_1 == matrix_2, true);
}

TEST_F(MatrixTest, CopyConstructorVecTest) {
    vector<vector<double>> data_1 = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    Matrix matrix_1(data_1);
    EXPECT_EQ(matrix_1.get_rows(), data_1.size());
    EXPECT_EQ(matrix_1.get_cols(), data_1[0].size());
    EXPECT_EQ(matrix_1 == data_1, true);
    vector<vector<double>> data_2 = { {1.0, 2.0}, {3.0, 4.0, 11.0, 12.0}, {5.0, 6.0, 7.0} };
    EXPECT_THROW(Matrix matrix_2(data_2), std::invalid_argument);
}

TEST_F(MatrixTest, MoveConstructorTest) {
    size_t rows = 5;
    size_t cols = 5;
    double val = 666.0;
    Matrix matrix_1(rows, cols, val);
    Matrix matrix_2(std::move(matrix_1));
    ASSERT_EQ(matrix_2.get_rows(), rows);
    ASSERT_EQ(matrix_2.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            EXPECT_NEAR(matrix_2[row][col], val, 1e-9);
        }
    }
    EXPECT_EQ(matrix_1.get_rows(), 0);
    EXPECT_EQ(matrix_1.get_cols(), 0);
    EXPECT_EQ(matrix_1.is_empty(), true);
}

TEST_F(MatrixTest, MoveConstructorVecTest) {
    size_t rows = 5;
    size_t cols = 5;
    vector<vector<double>> data_1(rows, vector<double>(5, 1.0));
    Matrix matrix_1(std::move(data_1));
    EXPECT_EQ(matrix_1.get_rows(), rows);
    EXPECT_EQ(matrix_1.get_cols(), cols);
    EXPECT_EQ(matrix_1 == vector<vector<double>>(rows, vector<double>(5, 1.0)), true);
    vector<vector<double>> data_2 = { {1.0, 2.0, 3.0}, {4.0, 5.0}, {6.0} };
    EXPECT_THROW(Matrix matrix_2(std::move(data_2)), std::invalid_argument);
}

TEST_F(MatrixTest, ConstructorInitListTest) {
    Matrix matrix_1 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    vector<vector<double>> result_1 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    EXPECT_EQ(matrix_1 == result_1, true);
    Matrix matrix_2 = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    vector<vector<double>> result_2 = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    EXPECT_EQ(matrix_2 == result_2, true);

    /* Lambda in parentheses is needed due to the EXPECT_THROW processes comma separated arguments */
    EXPECT_THROW(
        ( [] {
            Matrix matrix_3 = { {1.0, 2.0}, {3.0, 4.0, 5.0} };
        }() ),
        std::invalid_argument
    );
}

TEST_F(MatrixTest, CopyAssignmentOperatorMatrixTest) {
    Matrix matrix_1(5, 5, 666.0);
    Matrix matrix_2(1, 1, 1.0);
    matrix_2 = matrix_1;
    EXPECT_EQ(matrix_1.get_rows(), matrix_2.get_rows());
    EXPECT_EQ(matrix_1.get_cols(), matrix_2.get_cols());
    EXPECT_EQ(matrix_1 == matrix_2, true);
}

TEST_F(MatrixTest, MoveAssignmentOperatorMatrixTest) {
    size_t rows = 5;
    size_t cols = 5;
    double val = 666.0;
    Matrix matrix_1(rows, cols, val);
    Matrix matrix_2(1, 1, 1.0);
    matrix_2 = std::move(matrix_1);
    ASSERT_EQ(matrix_2.get_rows(), rows);
    ASSERT_EQ(matrix_2.get_cols(), cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            EXPECT_NEAR(matrix_2[row][col], val, 1e-9);
        }
    }
    EXPECT_EQ(matrix_1.get_rows(), 0);
    EXPECT_EQ(matrix_1.get_cols(), 0);
    EXPECT_EQ(matrix_1.is_empty(), true);
}

TEST_F(MatrixTest, CopyAssignmentOperatorVecTest) {
    vector<vector<double>> data_1 = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    Matrix matrix_1(1, 1, 1.0);
    matrix_1 = data_1;
    EXPECT_EQ(matrix_1.get_rows(), data_1.size());
    EXPECT_EQ(matrix_1.get_cols(), data_1[0].size());
    EXPECT_EQ(matrix_1 == data_1, true);
    vector<vector<double>> data_2 = { {1.0, 2.0}, {3.0, 4.0, 11.0, 12.0}, {5.0, 6.0, 7.0} };
    Matrix matrix_2(1, 1, 1.0);
    EXPECT_THROW(matrix_2 = data_2, std::invalid_argument);
}

TEST_F(MatrixTest, MoveAssignmentOperatorVecTest) {
    vector<vector<double>> data_1 = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    vector<vector<double>> result_1 = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    Matrix matrix_1(1, 1, 1.0);
    size_t rows = data_1.size();
    size_t cols = data_1[0].size();
    matrix_1 = std::move(data_1);
    EXPECT_EQ(matrix_1.get_rows(), rows);
    EXPECT_EQ(matrix_1.get_cols(), cols);
    EXPECT_EQ(matrix_1 == result_1, true);
    vector<vector<double>> data_2 = { {1.0, 2.0}, {3.0, 4.0, 11.0, 12.0}, {5.0, 6.0, 7.0} };
    Matrix matrix_2(1, 1, 1.0);
    EXPECT_THROW(matrix_2 = std::move(data_2), std::invalid_argument);
}

TEST_F(MatrixTest, AssignmentOperatorInitListTest) {
    Matrix matrix_1; 
    matrix_1 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    vector<vector<double>> result_1 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    EXPECT_EQ(matrix_1 == result_1, true);
    Matrix matrix_2; 
    matrix_2 = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    vector<vector<double>> result_2 = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    EXPECT_EQ(matrix_2 == result_2, true);

    /* Lambda in parentheses is needed due to the EXPECT_THROW processes comma separated arguments */
    EXPECT_THROW(
        ( [] {
            Matrix matrix_3;
            matrix_3 = { {1.0, 2.0}, {3.0, 4.0, 5.0} };
        }() ),
        std::invalid_argument
    );
}

TEST_F(MatrixTest, AccessOperatorTest) {
    Matrix matrix(5, 5);
    EXPECT_NEAR(matrix[2][2], 0.0, 1e-9);
    matrix[2][2] += 5.0;
    EXPECT_NEAR(matrix[2][2], 5.0, 1e-9);
    EXPECT_NEAR(matrix[1][1], 0.0, 1e-9);
}

TEST_F(MatrixTest, SecondaryAccessOperatorTest) {
    Matrix matrix(5, 5);
    EXPECT_THROW(matrix(5, 5), std::out_of_range);
    EXPECT_NEAR(matrix(2, 2), 0.0, 1e-9);
    matrix(2, 2) += 5.0;
    EXPECT_NEAR(matrix(2, 2), 5.0, 1e-9);
    EXPECT_NEAR(matrix(1, 1), 0.0, 1e-9);
}

TEST_F(MatrixTest, AtMethodTest) {
    Matrix matrix(5, 5);
    EXPECT_NEAR(matrix.at(2, 2), 0.0, 1e-9);
    matrix.at(2, 2) += 5.0;
    EXPECT_NEAR(matrix.at(2, 2), 5.0, 1e-9);
    EXPECT_THROW(matrix.at(-1, -1), std::out_of_range);
    EXPECT_THROW(matrix.at(5, 5), std::out_of_range);
    EXPECT_NO_THROW(matrix.at(0, 0));
    EXPECT_NO_THROW(matrix.at(4, 4));
}

TEST_F(MatrixTest, FillMethodTest) {
    Matrix matrix(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    matrix.fill(5.0);
    EXPECT_EQ(matrix == result, true);
}

TEST_F(MatrixTest, TransposeMethodTest) {
    vector<vector<double>> data = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    vector<vector<double>> result = { {1.0, 3.0, 5.0}, {2.0, 4.0, 6.0} };
    Matrix matrix(data);
    EXPECT_EQ(matrix.transpose() == result, true);
    EXPECT_EQ(matrix.transpose().transpose() == data, true);
    EXPECT_EQ(matrix.transpose().transpose() == matrix, true);
}

TEST_F(MatrixTest, ResizeMethodTest) {
    vector<vector<double>> data = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    vector<vector<double>> shrink_result = { {1.0}, {3.0} };
    vector<vector<double>> expand_result = { {1.0, 2.0, 0.0}, {3.0, 4.0, 0.0}, {5.0, 6.0, 0.0}, {0.0, 0.0, 0.0} };
    Matrix matrix(data);
    matrix.resize(shrink_result.size(), shrink_result[0].size());
    EXPECT_EQ(matrix == shrink_result, true);
    matrix = data;
    matrix.resize(expand_result.size(), expand_result[0].size());
    EXPECT_EQ(matrix == expand_result, true);
}

// TEST_F(MatrixTest, DeterminantMethodTest) {
//     Matrix matrix = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
//     EXPECT_NEAR(matrix.det(), 0.0, 1e-9);
//     matrix = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
//     EXPECT_NEAR(matrix.det(), -922020.0, 1e-9);
//     matrix = { {1.0} };
//     EXPECT_NEAR(matrix.det(), 1.0, 1e-9);
//     matrix = { {1.0, -90.999}, {2.555, -3.666} };
//     EXPECT_NEAR(matrix.det(), 228.836445, 1e-9);
// }

TEST_F(MatrixTest, AdditionWithValueTest) {
    Matrix matrix(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    EXPECT_EQ(matrix + 4.0 == result, true);
    EXPECT_EQ(matrix + 3.0 == result, false);
}

TEST_F(MatrixTest, AdditionWithMatrixTest) {
    Matrix matrix_1(5, 5, 1.0);
    Matrix matrix_2(5, 5, 4.0);
    Matrix matrix_3(4, 4, 4.0);
    Matrix result(5, 5, 5.0);
    EXPECT_THROW(matrix_1 + matrix_3, std::domain_error);
    EXPECT_EQ(matrix_1 + matrix_2 == result, true);
    EXPECT_EQ(matrix_2 + matrix_1 == result, true);
}

TEST_F(MatrixTest, SubtractionWithValueTest) {
    Matrix matrix(5, 5, 5.0);
    Matrix result(5, 5, 1.0);
    EXPECT_EQ(matrix - 4.0 == result, true);
    EXPECT_EQ(matrix - 3.0 == result, false);
}

TEST_F(MatrixTest, SubtractionWithMatrixTest) {
    Matrix matrix_1(5, 5, 5.0);
    Matrix matrix_2(5, 5, 1.0);
    Matrix matrix_3(4, 4, 1.0);
    Matrix result(5, 5, 4.0);
    EXPECT_THROW(matrix_1 - matrix_3, std::domain_error);
    EXPECT_EQ(matrix_1 - matrix_2 == result, true);
}

TEST_F(MatrixTest, MultiplyByValueTest) {
    Matrix matrix(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    EXPECT_EQ(matrix * 5 == result, true);
}

TEST_F(MatrixTest, MultiplyByValueFriendTest) {
    Matrix matrix(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    EXPECT_EQ(5 * matrix == result, true);
}

TEST_F(MatrixTest, MultiplyByMatrixTest) {
    Matrix matrix_1(2, 3, 1.0);
    Matrix matrix_2(3, 4, 2.0);
    EXPECT_THROW(matrix_2 * matrix_1, std::domain_error);
    Matrix result(matrix_1.get_rows(), matrix_2.get_cols(), 6.0);
    EXPECT_EQ(matrix_1 * matrix_2 == result, true);
}

TEST_F(MatrixTest, NegationOperatorTest) {
    Matrix matrix(5, 5, 1.0);
    Matrix result(5, 5, -1.0);
    EXPECT_EQ(-matrix == result, true);
}

TEST_F(MatrixTest, EqualsMethodMat) {
    Matrix matrix_1(5, 5, 6.66);
    Matrix matrix_2(5, 5, 6.66);
    Matrix matrix_3(4, 4, 6.66);
    Matrix matrix_4(5, 5, 1.00);
    EXPECT_EQ(matrix_1.equals(matrix_2), true);
    EXPECT_EQ(matrix_2.equals(matrix_3), false);
    EXPECT_EQ(matrix_3.equals(matrix_4), false);
    EXPECT_EQ(matrix_4.equals(matrix_1), false);
}

TEST_F(MatrixTest, EqualsMethodVec) {
    Matrix matrix(3, 3, 5.0);
    vector<vector<double>> data_1 = { {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} };
    vector<vector<double>> data_2 = { {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0} };
    vector<vector<double>> data_3 = { {5.0, 5.0}, {5.0, 5.0} };
    vector<vector<double>> data_4 = {};
    EXPECT_EQ(matrix.equals(data_1), true);
    EXPECT_EQ(matrix.equals(data_2), false);
    EXPECT_EQ(matrix.equals(data_3), false);
    EXPECT_EQ(matrix.equals(data_4), false);
}

TEST_F(MatrixTest, IdentityMethodTest) {
    Matrix matrix = Matrix::identity(5);
    EXPECT_EQ(matrix.get_rows() == 5, true);
    EXPECT_EQ(matrix.get_cols() == 5, true);
    for (size_t row = 0; row < matrix.get_rows(); ++row) {
        for (size_t col = 0; col < matrix.get_cols(); ++col) {
            if (row == col) {
                EXPECT_NEAR(matrix[row][col], 1.0, 1e-9);
            } else {
                EXPECT_NEAR(matrix[row][col], 0.0, 1e-9);
            }
        }
    }
}

TEST_F(MatrixTest, ZerosMethodTest) {
    size_t rows = 5;
    size_t cols = 7;
    Matrix matrix = Matrix::zeros(rows, cols);
    EXPECT_EQ(matrix.get_rows() == rows, true);    
    EXPECT_EQ(matrix.get_cols() == cols, true);
    for (size_t row = 0; row < matrix.get_rows(); ++row) {
        for (size_t col = 0; col < matrix.get_cols(); ++col) {
            EXPECT_NEAR(matrix[row][col], 0.0, 1e-9);
        }
    }
}

TEST_F(MatrixTest, OnesMethodTest) {
    size_t rows = 5;
    size_t cols = 7;
    Matrix matrix = Matrix::ones(rows, cols);
    EXPECT_EQ(matrix.get_rows() == rows, true);    
    EXPECT_EQ(matrix.get_cols() == cols, true);
    for (size_t row = 0; row < matrix.get_rows(); ++row) {
        for (size_t col = 0; col < matrix.get_cols(); ++col) {
            EXPECT_NEAR(matrix[row][col], 1.0, 1e-9);
        }
    }
}

