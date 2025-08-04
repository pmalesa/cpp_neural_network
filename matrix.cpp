#include "matrix.h"

Matrix::Matrix(size_t rows, size_t cols)
    : data_(vector<vector<double>>(rows, vector<double>(cols, 0.0))), rows_(rows), cols_(cols) {}

Matrix::Matrix(const Matrix& mat)
    : data_(mat.data_), rows_(mat.rows_), cols_(mat.cols_) {} 

Matrix::Matrix(Matrix&& mat) noexcept
    : data_(move(mat.data_)), rows_(mat.rows_), cols_(mat.cols_) {}

Matrix::~Matrix() = default;

double Matrix::at(size_t row, size_t col) const {
    return data_[row][col];
}