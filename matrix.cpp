#include "matrix.h"
#include <stdexcept>
#include <iostream>

Matrix::Matrix(size_t rows, size_t cols)
    : data_(vector<vector<double>>(rows, vector<double>(cols, 0.0))), rows_(rows), cols_(cols) {}

Matrix::Matrix(size_t rows, size_t cols, double val) 
    : data_(vector<vector<double>>(rows, vector<double>(cols, val))), rows_(rows), cols_(cols) {}

Matrix::Matrix(const Matrix& mat)
    : data_(mat.data_), rows_(mat.rows_), cols_(mat.cols_) {} 

Matrix::Matrix(const vector<vector<double>>& data)
    : data_(data), rows_(data.size()), cols_(0) {
    if (!data.empty()) {
        cols_ = data[0].size();
        for (const auto& row : data) {
            if (row.size() != cols_) {
                throw std::invalid_argument("All rows must have the same number of columns!");
            }
        }
    }
}

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

double& Matrix::operator()(size_t row, size_t col) {
    if (row >= this->rows_ || row < 0 || col >= this->cols_ || col < 0) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return this->data_[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= this->rows_ || row < 0 || col >= this->cols_ || col < 0) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return this->data_[row][col];
}

Matrix& Matrix::operator=(const Matrix& mat) {
    if (this != &mat) {
        data_ = mat.data_;
        rows_ = mat.rows_;
        cols_ = mat.cols_;    
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& mat) noexcept {
    if (this != &mat) {
        data_ = std::move(mat.data_);
        rows_ = mat.rows_;
        cols_ = mat.cols_;
        mat.rows_ = 0;
        mat.cols_ = 0;
    }
    return *this;
}

Matrix& Matrix::operator=(const vector<vector<double>>& data) {
    if (!data.empty()) {
        size_t cols = data[0].size();
        for (const auto& row : data) {
            if (row.size() != cols) {
                throw std::invalid_argument("All rows must have the same number of columns!");
            }
        }
        rows_ = data.size();
        cols_ = cols;
    } else {
        rows_ = 0;
        cols_ = 0;
    }
    data_ = data;
    return *this;
}

Matrix& Matrix::operator=(vector<vector<double>>&& data) {
    if (!data.empty()) {
        size_t cols = data[0].size();
        for (const auto& row : data) {
            if (row.size() != cols) {
                throw std::invalid_argument("All rows must have the same number of columns!");
            }
        }
        rows_ = data.size();
        cols_ = cols;
    } else {
        rows_ = 0;
        cols_ = 0;
    }
    data_ = std::move(data);
    return *this;
}

bool Matrix::operator==(const Matrix& mat) const {
    if (this->rows_ != mat.rows_ || this->cols_ != mat.cols_) {
        return false;
    }
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            if (abs(this->data_[row][col] - mat.data_[row][col]) > 1e-9) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator==(const vector<vector<double>>& data) const {
    if (data.empty()) {
        return this->data_.empty();
    }
    if (this->rows_ != data.size() || this->cols_ != data[0].size()) {
        return false;
    }
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            if (abs(this->data_[row][col] - data[row][col]) > 1e-9) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix& mat) const {
    if (this->rows_ != mat.rows_ || this->cols_ != mat.cols_) {
        return true;
    }
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            if (abs(this->data_[row][col] - mat.data_[row][col]) > 1e-9) {
                return true;
            }
        }
    }
    return false;
}

bool Matrix::operator!=(const vector<vector<double>>& data) const {
    if (data.empty()) {
        return !this->data_.empty();
    }
    if (this->rows_ != data.size() || this->cols_ != data[0].size()) {
        return true;
    }
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            if (abs(this->data_[row][col] - data[row][col]) > 1e-9) {
                return true;
            }
        }
    }
    return false;
}

Matrix Matrix::operator+(double val) const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = this->data_[row][col] + val;
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& mat) const {
    if (this->rows_ != mat.rows_ || this->cols_ != mat.cols_) {
        throw std::domain_error("Matrices can not be added!");
    }
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = this->data_[row][col] + mat.data_[row][col];
        }
    }
    return result;
}

Matrix Matrix::operator-(double val) const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = this->data_[row][col] - val;
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& mat) const {
    if (this->rows_ != mat.rows_ || this->cols_ != mat.cols_) {
        throw std::domain_error("Matrices can not be added!");
    }
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = this->data_[row][col] - mat.data_[row][col];
        }
    }
    return result;
}

Matrix Matrix::operator*(double val) const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = this->data_[row][col] * val;
        }
    }
    return result;
}

Matrix operator*(double val, const Matrix& mat) {
    Matrix result(mat.get_rows(), mat.get_cols());
    for (size_t row = 0; row < mat.get_rows(); ++row) {
        for (size_t col = 0; col < mat.get_cols(); ++col) {
            result[row][col] = mat.data_[row][col] * val;
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& mat) const {
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

Matrix Matrix::operator-() const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            result[row][col] = -this->data_[row][col];
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

void Matrix::fill(double val) {
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            this->data_[row][col] = val;
        }
    }
}

Matrix Matrix::transpose() const {
    Matrix mat(this->cols_, this->rows_);
    for (size_t i = 0; i < this->cols_; ++i) {
        for (size_t j = 0; j < this->rows_; ++j) {
            mat[i][j] = this->data_[j][i];
        }
    }
    return mat;
}

bool Matrix::equals(const Matrix& mat, double epsilon) const {
    if (this->rows_ != mat.rows_ || this->cols_ != mat.cols_) {
        return false;
    }
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            if (abs(this->data_[row][col] - mat.data_[row][col]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::equals(const vector<vector<double>>& data, double epsilon) const {
    if (data.empty()) {
        return this->data_.empty();
    }
    if (this->rows_ != data.size() || this->cols_ != data[0].size()) {
        return false;
    }
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            if (abs(this->data_[row][col] - data[row][col]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

ostream& operator<<(ostream& os, const Matrix& mat) {
    os << "\n[";
    for (size_t row = 0; row < mat.get_rows(); ++row) {
        os << "[";
        for (size_t col = 0; col < mat.get_cols(); ++col) {
            os << mat.data_[row][col];
            if (col < mat.get_cols() - 1) {
                os << ", ";
            }
        }
        os << "]";
        if (row < mat.get_rows() - 1) {
            os << ",\n";
        }
    }
    os << "]\n";
    return os;
}

