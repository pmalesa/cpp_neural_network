#include "matrix.h"
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols)
    : data_(vector<vector<double>>(rows, vector<double>(cols, 0.0))), rows_(rows), cols_(cols) {}

Matrix::Matrix(const Matrix& mat)
    : data_(mat.data_), rows_(mat.rows_), cols_(mat.cols_) {} 

Matrix::Matrix(Matrix&& mat) noexcept
    : data_(move(mat.data_)), rows_(mat.rows_), cols_(mat.cols_) {
    mat.rows_ = 0;
    mat.cols_ = 0;
}

vector<double>& Matrix::operator[](size_t row) {
    return this->data_[row];
}

const vector<double>& Matrix::operator[](size_t row) const {
    return this->data_[row];
}

double& Matrix::at(size_t row, size_t col) {
    if (row >= this->rows_ || col >= this->cols_) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return data_[row][col];
}

const double& Matrix::at(size_t row, size_t col) const {
    if (row >= this->rows_ || col >= this->cols_) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return data_[row][col];
}

