#include "matrix.h"
#include <gtest/gtest.h>
#include <vector>
#include <limits>

using std::vector;
using std::initializer_list;
using il = std::initializer_list<std::initializer_list<double>>;

#define DOUBLE_MAX (std::numeric_limits<double>::max())
#define DOUBLE_MIN (std::numeric_limits<double>::lowest())

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

TEST_F(MatrixTest, EqualsOperatorInitList) {
    vector<vector<double>> data_1 = { {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} };
    vector<vector<double>> data_2 = { {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} };
    vector<vector<double>> data_3 = { {1.0} };
    vector<vector<double>> data_4 = { {} };
    vector<vector<double>> data_5 = {};
    Matrix matrix_1(data_1);
    Matrix matrix_2(data_2);
    Matrix matrix_3(data_3);
    Matrix matrix_4(data_4);
    Matrix matrix_5(data_5);

    EXPECT_TRUE((matrix_1 == il{ {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} }));
    EXPECT_TRUE((matrix_2 == il{ {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} }));
    EXPECT_TRUE((matrix_3 == il{ {1.0} }));
    EXPECT_TRUE((matrix_4 == il{ {} }));
    EXPECT_TRUE((matrix_5 == il{ }));
    EXPECT_TRUE((il{ {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} } == matrix_1));
    EXPECT_TRUE((il{ {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} } == matrix_2));
    EXPECT_TRUE((il{ {1.0} } == matrix_3));
    EXPECT_TRUE((il{ {} } == matrix_4));
    EXPECT_TRUE((il{ } == matrix_5));
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

TEST_F(MatrixTest, NotEqualsOperatorInitList) {
    vector<vector<double>> data_1 = { {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} };
    vector<vector<double>> data_2 = { {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} };
    vector<vector<double>> data_3 = { {1.0} };
    vector<vector<double>> data_4 = { {} };
    vector<vector<double>> data_5 = {};
    Matrix matrix_1(data_1);
    Matrix matrix_2(data_2);
    Matrix matrix_3(data_3);
    Matrix matrix_4(data_4);
    Matrix matrix_5(data_5);

    EXPECT_FALSE((matrix_1 != il{ {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} }));
    EXPECT_FALSE((matrix_2 != il{ {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} }));
    EXPECT_FALSE((matrix_3 != il{ {1.0} }));
    EXPECT_FALSE((matrix_4 != il{ {} }));
    EXPECT_FALSE((matrix_5 != il{ }));
    EXPECT_FALSE((il{ {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} } != matrix_1));
    EXPECT_FALSE((il{ {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} } != matrix_2));
    EXPECT_FALSE((il{ {1.0} } != matrix_3));
    EXPECT_FALSE((il{ {} } != matrix_4));
    EXPECT_FALSE((il{ } != matrix_5));
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
    vector<vector<double>> data_3 = { {} };
    Matrix matrix_3(data_3);
    EXPECT_TRUE(matrix_3.get_rows() == 1);
    EXPECT_TRUE(matrix_3.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    vector<vector<double>> data_4 = { };
    Matrix matrix_4(data_4);
    EXPECT_TRUE(matrix_4.get_rows() == 0);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    EXPECT_TRUE(matrix_4.is_empty());
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
    vector<vector<double>> data_3 = { {} };
    Matrix matrix_3(std::move(data_3));
    EXPECT_TRUE(matrix_3.get_rows() == 1);
    EXPECT_TRUE(matrix_3.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    vector<vector<double>> data_4 = { };
    Matrix matrix_4(std::move(data_4));
    EXPECT_TRUE(matrix_4.get_rows() == 0);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    EXPECT_TRUE(matrix_4.is_empty());
}

TEST_F(MatrixTest, ConstructorInitListTest) {
    Matrix matrix_1 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    EXPECT_TRUE((matrix_1 == il{ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} }));
    Matrix matrix_2 = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    EXPECT_TRUE((matrix_2 == il{ {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} }));

    /* Lambda in parentheses is needed due to the EXPECT_THROW processes comma separated arguments */
    EXPECT_THROW(
        ( [] {
            Matrix matrix_3 = { {1.0, 2.0}, {3.0, 4.0, 5.0} };
        }() ),
        std::invalid_argument
    );
    Matrix matrix_3 = { {} };
    EXPECT_TRUE(matrix_3.get_rows() == 1);
    EXPECT_TRUE(matrix_3.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    Matrix matrix_4 = { };
    EXPECT_TRUE(matrix_4.get_rows() == 0);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    EXPECT_TRUE(matrix_4.is_empty());
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
    vector<vector<double>> data_3 = { {} };
    Matrix matrix_3;
    matrix_3 = data_3;
    EXPECT_TRUE(matrix_3.get_rows() == 1);
    EXPECT_TRUE(matrix_3.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    vector<vector<double>> data_4 = { };
    Matrix matrix_4;
    matrix_4 = data_4;
    EXPECT_TRUE(matrix_4.get_rows() == 0);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    EXPECT_TRUE(matrix_4.is_empty());
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
    vector<vector<double>> data_3 = { {} };
    Matrix matrix_3;
    matrix_3 = std::move(data_3);
    EXPECT_TRUE(matrix_3.get_rows() == 1);
    EXPECT_TRUE(matrix_3.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    vector<vector<double>> data_4 = { };
    Matrix matrix_4;
    matrix_4 = std::move(data_4);
    EXPECT_TRUE(matrix_4.get_rows() == 0);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    EXPECT_TRUE(matrix_4.is_empty());
}

TEST_F(MatrixTest, AssignmentOperatorInitListTest) {
    Matrix matrix_1; 
    matrix_1 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    EXPECT_TRUE((matrix_1 == il{ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} }));
    Matrix matrix_2; 
    matrix_2 = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    EXPECT_TRUE((matrix_2 == il{ {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} }));

    /* Lambda in parentheses is needed due to the EXPECT_THROW processes comma separated arguments */
    EXPECT_THROW(
        ( [] {
            Matrix matrix_3;
            matrix_3 = { {1.0, 2.0}, {3.0, 4.0, 5.0} };
        }() ),
        std::invalid_argument
    );

    Matrix matrix_3;
    matrix_3 = { {} };
    EXPECT_TRUE(matrix_3.get_rows() == 1);
    EXPECT_TRUE(matrix_3.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    Matrix matrix_4;
    matrix_4 = { };
    EXPECT_TRUE(matrix_4.get_rows() == 0);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    EXPECT_TRUE(matrix_4.is_empty());
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

TEST_F(MatrixTest, DeterminantMethodTest) {
    Matrix matrix = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
    EXPECT_NEAR(matrix.det(), 0.0, 1e-9);
    matrix = { {1.0, -2.0, -3.0, 123.0}, {-4.0, 0.0, -6.0, 7.0}, {-7.0, 8.0, 15.0, -9.0}, {-70.0, 80.0, -90.0, -11.0} };
    EXPECT_NEAR(matrix.det(), -922020.0, 1e-9);
    matrix = { {1.0} };
    EXPECT_NEAR(matrix.det(), 1.0, 1e-9);
    matrix = { {1.0, -90.999}, {2.555, -3.666} };
    EXPECT_NEAR(matrix.det(), 228.836445, 1e-9);
    matrix = { };
    EXPECT_NEAR(matrix.det(), 1.0, 1e-9);
    matrix = Matrix(0, 0);
    EXPECT_NEAR(matrix.det(), 1.0, 1e-9);
    matrix = Matrix(0, 1, 1.0);
    EXPECT_THROW(matrix.det(), std::domain_error);
    matrix = Matrix(1, 0, 1.0);
    EXPECT_THROW(matrix.det(), std::domain_error);
    matrix = { {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} };
    EXPECT_THROW(matrix.det(), std::domain_error);
}

TEST_F(MatrixTest, InverseMethodTest){
    Matrix matrix_1 = {};
    Matrix result_1 = {};
    EXPECT_TRUE(matrix_1.inverse() == result_1);
    EXPECT_TRUE(matrix_1.inverse().inverse() == matrix_1);    

    Matrix matrix_2 = { {2.0} };
    Matrix result_2 = { {0.5} };
    EXPECT_TRUE(matrix_2.inverse() == result_2);
    EXPECT_TRUE(matrix_2.inverse().inverse() == matrix_2);    

    Matrix matrix_3 = { {2.0, 3.0}, {4.0, 5.0} };
    Matrix result_3 = { {-2.5, 1.5}, {2.0, -1.0} };
    EXPECT_TRUE(matrix_3.inverse() == result_3);
    EXPECT_TRUE(matrix_3.inverse().inverse() == matrix_3);    

    Matrix matrix_4 = { {1.0, -6.0, 4.0}, {2.0, -2.0, 2.0}, {-3.0, -5.0, 7.0} };
    Matrix result_4 = {
        {-0.07692307692307692308, 0.42307692307692307693, -0.076923076923076923076},
        {-0.38461538461538461538, 0.36538461538461538461, 0.11538461538461538461},
        {-0.3076923076923076923, 0.44230769230769230769, 0.19230769230769230769}
    };
    EXPECT_TRUE(matrix_4.inverse() == result_4);
    EXPECT_TRUE(matrix_4.inverse().inverse() == matrix_4);    

    Matrix matrix_5 = { {1.5, 2.5, 7.9, -1.2}, {-4.5, -2.5, 4.0, 4.0}, {-11.2, -11.3, -1.5, 1.5}, {6.6, 7.7, -8.8, -9.9} };;
    Matrix result_5 = {
        {0.0132532296447390787, -0.41316611422743498215, -0.02286779704232534421, -0.172007073540092408},
        {-0.04650582185959544435, 0.42776325854156042831, -0.08007712901580826109, 0.16633781079535796514},
        {0.11861507734149243581, -0.042484701682814890354, 0.015776389597144314124, -0.029152759105589294264},
        {-0.132771332653408125, 0.0950248597654258031, -0.091550866904640489536, -0.060394066879915936542}
    };
    EXPECT_TRUE(matrix_5.inverse() == result_5);
    EXPECT_TRUE(matrix_5.inverse().inverse() == matrix_5);    

    Matrix matrix_6(2, 3, 5.0); // rows != cols
    EXPECT_THROW(matrix_6.inverse(), std::invalid_argument);
    Matrix matrix_7(2, 2, 1.0); // det() == 0
    EXPECT_THROW(matrix_7.inverse(), std::runtime_error);
    Matrix matrix_8(1, 1, 0.0); // singular matrix with 0.0
    EXPECT_THROW(matrix_8.inverse(), std::runtime_error);
}

TEST_F(MatrixTest, AdditionWithValueTest) {
    Matrix matrix_1(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    EXPECT_EQ(matrix_1 + 4.0 == result, true);
    EXPECT_EQ(matrix_1 + 3.0 == result, false);
    Matrix matrix_2(5, 5, DOUBLE_MAX);
    EXPECT_THROW(matrix_2 + DOUBLE_MAX, std::overflow_error);
}

TEST_F(MatrixTest, AdditionWithMatrixTest) {
    Matrix matrix_1(5, 5, 1.0);
    Matrix matrix_2(5, 5, 4.0);
    Matrix matrix_3(4, 4, 4.0);
    Matrix result(5, 5, 5.0);
    EXPECT_THROW(matrix_1 + matrix_3, std::domain_error);
    EXPECT_EQ(matrix_1 + matrix_2 == result, true);
    EXPECT_EQ(matrix_2 + matrix_1 == result, true);
    Matrix matrix_4(5, 5, DOUBLE_MAX);
    EXPECT_THROW(matrix_4 + matrix_4, std::overflow_error);
}

TEST_F(MatrixTest, SubtractionWithValueTest) {
    Matrix matrix_1(5, 5, 5.0);
    Matrix result(5, 5, 1.0);
    EXPECT_EQ(matrix_1 - 4.0 == result, true);
    EXPECT_EQ(matrix_1 - 3.0 == result, false);
    Matrix matrix_2(5, 5, DOUBLE_MIN);
    EXPECT_THROW(matrix_2 - DOUBLE_MAX, std::overflow_error);
}

TEST_F(MatrixTest, SubtractionWithMatrixTest) {
    Matrix matrix_1(5, 5, 5.0);
    Matrix matrix_2(5, 5, 1.0);
    Matrix matrix_3(4, 4, 1.0);
    Matrix result(5, 5, 4.0);
    EXPECT_THROW(matrix_1 - matrix_3, std::domain_error);
    EXPECT_EQ(matrix_1 - matrix_2 == result, true);
    Matrix matrix_4(5, 5, DOUBLE_MIN);
    Matrix matrix_5(5, 5, DOUBLE_MAX);
    EXPECT_THROW(matrix_4 - matrix_5, std::overflow_error);
    EXPECT_THROW(matrix_4 - matrix_5, std::overflow_error);
}

TEST_F(MatrixTest, MultiplyByValueTest) {
    Matrix matrix_1(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    EXPECT_EQ(matrix_1 * 5 == result, true);
    Matrix matrix_2(5, 5, DOUBLE_MAX);
    Matrix matrix_3(5, 5, DOUBLE_MIN);
    EXPECT_THROW(matrix_2 * DOUBLE_MAX, std::overflow_error);
    EXPECT_THROW(matrix_2 * DOUBLE_MIN, std::overflow_error);
    EXPECT_THROW(matrix_2 * 4.0, std::overflow_error);
    EXPECT_THROW(matrix_2 * -4.0, std::overflow_error);
    EXPECT_THROW(matrix_3 * DOUBLE_MAX, std::overflow_error);
    EXPECT_THROW(matrix_3 * DOUBLE_MIN, std::overflow_error);
    EXPECT_THROW(matrix_3 * 4.0, std::overflow_error);
    EXPECT_THROW(matrix_3 * -4.0, std::overflow_error);
}

TEST_F(MatrixTest, MultiplyByValueFriendTest) {
    Matrix matrix_1(5, 5, 1.0);
    Matrix result(5, 5, 5.0);
    EXPECT_EQ(5 * matrix_1 == result, true);
    Matrix matrix_2(5, 5, DOUBLE_MAX);
    Matrix matrix_3(5, 5, DOUBLE_MIN);
    EXPECT_THROW(DOUBLE_MAX * matrix_2, std::overflow_error);
    EXPECT_THROW(DOUBLE_MIN * matrix_2, std::overflow_error);
    EXPECT_THROW(4.0 * matrix_2, std::overflow_error);
    EXPECT_THROW(-4.0 * matrix_2, std::overflow_error);
    EXPECT_THROW(DOUBLE_MAX * matrix_3, std::overflow_error);
    EXPECT_THROW(DOUBLE_MIN * matrix_3, std::overflow_error);
    EXPECT_THROW(4.0 * matrix_3, std::overflow_error);
    EXPECT_THROW(-4.0 * matrix_3, std::overflow_error);
}

TEST_F(MatrixTest, MultiplyByMatrixTest) {
    Matrix matrix_1(2, 3, 1.0);
    Matrix matrix_2(3, 4, 2.0);
    EXPECT_THROW(matrix_2 * matrix_1, std::domain_error);
    Matrix result_1(matrix_1.get_rows(), matrix_2.get_cols(), 6.0);
    EXPECT_EQ(matrix_1 * matrix_2 == result_1, true);
    Matrix matrix_3(5, 5, DOUBLE_MAX);
    Matrix matrix_4(5, 5, DOUBLE_MIN);
    EXPECT_THROW(matrix_3 * matrix_3, std::overflow_error);
    EXPECT_THROW(matrix_3 * matrix_4, std::overflow_error);
    EXPECT_THROW(matrix_3 * Matrix(5, 5, 4.0), std::overflow_error);
    EXPECT_THROW(matrix_3 * Matrix(5, 5, -4.0), std::overflow_error);
    EXPECT_THROW(matrix_4 * matrix_3, std::overflow_error);
    EXPECT_THROW(matrix_4 * matrix_4, std::overflow_error);
    EXPECT_THROW(matrix_4 * Matrix(5, 5, 4.0), std::overflow_error);
    EXPECT_THROW(matrix_4 * Matrix(5, 5, -4.0), std::overflow_error);

    // Concurrent computation
    Matrix matrix_5(500, 500, 2.0);
    Matrix matrix_6(500, 500, 3.0);
    Matrix result_2(500, 500, 3000.0);
    EXPECT_TRUE(matrix_5 * matrix_6 == result_2);
}

TEST_F(MatrixTest, NegationOperatorTest) {
    Matrix matrix(5, 5, 1.0);
    Matrix result(5, 5, -1.0);
    EXPECT_EQ(-matrix == result, true);
}

TEST_F(MatrixTest, IsEmptyMethodTest) {
    Matrix matrix_1(0, 5);
    Matrix matrix_2(5, 0);
    Matrix matrix_3(0, 4, 6.66);
    Matrix matrix_4(5, 0, 1.00);
    EXPECT_TRUE(matrix_1.is_empty());
    EXPECT_TRUE(matrix_1.get_rows() == 0);
    EXPECT_TRUE(matrix_1.get_cols() == 5);
    EXPECT_TRUE(matrix_2.is_empty());
    EXPECT_TRUE(matrix_2.get_rows() == 5);
    EXPECT_TRUE(matrix_2.get_cols() == 0);
    EXPECT_TRUE(matrix_3.is_empty());
    EXPECT_TRUE(matrix_3.get_rows() == 0);
    EXPECT_TRUE(matrix_3.get_cols() == 4);
    EXPECT_TRUE(matrix_4.is_empty());
    EXPECT_TRUE(matrix_4.get_rows() == 5);
    EXPECT_TRUE(matrix_4.get_cols() == 0);
    Matrix matrix_5(5, 5);
    Matrix matrix_6(5, 5);
    Matrix matrix_7(4, 4, 6.66);
    Matrix matrix_8(5, 5, 1.00);
    EXPECT_FALSE(matrix_5.is_empty());
    EXPECT_FALSE(matrix_6.is_empty());
    EXPECT_FALSE(matrix_7.is_empty());
    EXPECT_FALSE(matrix_8.is_empty());
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

TEST_F(MatrixTest, EqualsMethodInitList) {
    vector<vector<double>> data_1 = { {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} };
    vector<vector<double>> data_2 = { {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} };
    vector<vector<double>> data_3 = { {1.0} };
    vector<vector<double>> data_4 = { {} };
    vector<vector<double>> data_5 = {};
    Matrix matrix_1(data_1);
    Matrix matrix_2(data_2);
    Matrix matrix_3(data_3);
    Matrix matrix_4(data_4);
    Matrix matrix_5(data_5);

    EXPECT_TRUE((matrix_1 == il{ {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} }));
    EXPECT_TRUE((matrix_2 == il{ {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} }));
    EXPECT_TRUE((matrix_3 == il{ {1.0} }));
    EXPECT_TRUE((matrix_4 == il{ {} }));
    EXPECT_TRUE((matrix_5 == il{ }));
    EXPECT_TRUE((il{ {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0}, {5.0, 5.0, 5.0} } == matrix_1));
    EXPECT_TRUE((il{ {1.777, 2.66}, {1.11, 2.222}, {1.0, 2.333} } == matrix_2));
    EXPECT_TRUE((il{ {1.0} } == matrix_3));
    EXPECT_TRUE((il{ {} } == matrix_4));
    EXPECT_TRUE((il{ } == matrix_5));
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

