#include "matrix.h"
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols)
    : data_(vector<vector<double>>(rows, vector<double>(cols, 0.0))), rows_(rows), cols_(cols) {}

Matrix::Matrix(size_t rows, size_t cols, double val) 
    : data_(vector<vector<double>>(rows, vector<double>(cols, val))), rows_(rows), cols_(cols) {}

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

Matrix Matrix::operator*(double val) {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = this->data_[row][col] * val;
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& mat) {
    if (this->cols_ != mat.rows_) {
        throw std::domain_error("Matrices can not be multiplied!");
    }
    Matrix result(this->rows_, mat.cols_);
    size_t common_dim = this->cols_; 
    for (size_t row = 0; row < result.rows_; ++row) {
        for (size_t col = 0; col < result.cols_; ++col) {
            double sum = 0;
            for (size_t i = 0; i < common_dim; ++i) {
                sum += this->data_[row][i] * mat.data_[i][col];
            }
            result[row][col] = sum;
        }
    }
    return result;
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

