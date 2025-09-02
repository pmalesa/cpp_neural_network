#include "matrix.h"
#include <stdexcept>
#include <cmath>

Matrix::Matrix(size_t rows, size_t cols)
    : data_(vector<vector<double>>(rows, vector<double>(cols, 0.0))), rows_(rows), cols_(cols) {}

Matrix::Matrix(size_t rows, size_t cols, double val) 
    : data_(vector<vector<double>>(rows, vector<double>(cols, val))), rows_(rows), cols_(cols) {}

Matrix::Matrix(const Matrix& mat)
    : data_(mat.data_), rows_(mat.rows_), cols_(mat.cols_) {} 

Matrix::Matrix(const vector<vector<double>>& data) {
    if (!data.empty()) {
        size_t cols = data[0].size();
        cols_ = data[0].size();
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
}

Matrix::Matrix(Matrix&& mat) noexcept
    : data_(move(mat.data_)), rows_(mat.rows_), cols_(mat.cols_) {
    mat.rows_ = 0;
    mat.cols_ = 0;
}

Matrix::Matrix(vector<vector<double>>&& data) {
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
    data_ = move(data);
}

Matrix::Matrix(initializer_list<initializer_list<double>> init_list)
    : rows_(init_list.size()), cols_(0) {
    if (rows_ == 0) {
        return;
    }
    cols_ = init_list.begin()->size();
    for (const auto& row : init_list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns!");
        }
    }

    data_.reserve(rows_);
    for (const auto& row : init_list) {
        data_.emplace_back(row);
    }
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

Matrix& Matrix::operator=(initializer_list<initializer_list<double>> init_list) {
    if (init_list.size() == 0) {
        rows_ = 0;
        cols_ = 0;
        data_.clear();
        return *this;
    }
    size_t cols = init_list.begin()->size();
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns!");
        }
    }

    rows_ = init_list.size();
    cols_ = cols;
    
    data_.clear();
    data_.reserve(rows_);
    for (const auto& row : init_list) {
        data_.emplace_back(row);
    }

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

bool Matrix::operator==(initializer_list<initializer_list<double>> init_list) const {
    if (init_list.size() != rows_) {
        return false;
    }

    // Case of empty init list and 0 rows
    if (rows_ == 0) {
        return cols_ == 0;
    }

    const size_t cols = init_list.begin()->size();
    if (cols != cols_) {
        return false;
    }

    size_t i = 0;
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            return false;
        }
        size_t j = 0;
        for (double val : row) {
            if (abs(data_[i][j] - val) > eps) {
                return false;
            }
            ++j;
        }
        ++i;
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

bool Matrix::operator!=(initializer_list<initializer_list<double>> init_list) const {
    if (init_list.size() != rows_) {
        return true;
    }

    // Case of empty init list and 0 rows
    if (rows_ == 0) {
        return cols_ != 0;
    }

    const size_t cols = init_list.begin()->size();
    if (cols != cols_) {
        return true;
    }

    size_t i = 0;
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            return true;
        }
        size_t j = 0;
        for (double val : row) {
            if (abs(data_[i][j] - val) > eps) {
                return true;
            }
            ++j;
        }
        ++i;
    }
    
    return false;
}

Matrix Matrix::operator+(double val) const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            double v = this->data_[row][col] + val;
            if (!std::isfinite(v)) {
                throw std::overflow_error("Addition overflowed!");
            }
            result[row][col] = v;
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
            double v = this->data_[row][col] + mat.data_[row][col];
            if (!std::isfinite(v)) {
                throw std::overflow_error("Addition overflowed!");
            }
            result[row][col] = v;
        }
    }
    return result;
}

Matrix Matrix::operator-(double val) const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        for (size_t col = 0; col < this->cols_; ++col) {
            double v = this->data_[row][col] - val;
            if (!std::isfinite(v)) {
                throw std::overflow_error("Subtraction overflowed!");
            }
            result[row][col] = v;
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
            double v = this->data_[row][col] - mat.data_[row][col];
            if (!std::isfinite(v)) {
                throw std::overflow_error("Subtraction overflowed!");
            }
            result[row][col] = v;
        }
    }
    return result;
}

Matrix Matrix::operator*(double val) const {
    Matrix result(this->rows_, this->cols_);
    for (size_t row = 0; row < this->rows_; ++row) {
        long double v_ld = 0.0;
        for (size_t col = 0; col < this->cols_; ++col) {
            v_ld = static_cast<long double>(this->data_[row][col]) * 
                   static_cast<long double>(val);
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Matrix multiplication overflowed!");
            }
            result[row][col] = v_d;
        }
    }
    return result;
}

Matrix operator*(double val, const Matrix& mat) {
    Matrix result(mat.get_rows(), mat.get_cols());
    for (size_t row = 0; row < mat.get_rows(); ++row) {
        long double v_ld = 0.0;
        for (size_t col = 0; col < mat.get_cols(); ++col) {
            v_ld = static_cast<long double>(mat.data_[row][col]) * 
                   static_cast<long double>(val);
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Matrix multiplication overflowed!");
            }
            result[row][col] = v_d;
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
            long double v_ld = 0.0;
            for (size_t i = 0; i < common_dim; ++i) {
                v_ld += static_cast<long double>(this->data_[row][i]) * 
                       static_cast<long double>(mat.data_[i][col]);
            }
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Matrix multiplication overflowed!");
            }
            result[row][col] = v_d;
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

void Matrix::resize(size_t new_rows, size_t new_cols) {
    data_.resize(new_rows);
    for (auto& row : data_) {
        row.resize(new_cols, 0.0);
    }
    rows_ = new_rows;
    cols_ = new_cols;
}

double Matrix::det() const {
    if (rows_ != cols_) {
        throw std::domain_error("Matrix has to be square in order to calculate determinant!");
    }
    size_t n = rows_;
    if (n == 0) {
        return 1.0;
    } 
    if (n == 1) {
        return data_[0][0];
    }
    if (n == 2) {
        return data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0];
    }

    auto data = data_;
    double det = 1.0;
    int sign = 1;

    /* 
    Gaussian elimination method - O(N^3):
        1) We need to apply the row operations to transform a square matrix into an upper triangular
           matrix (all zeros below the diagonal).
        2) For an upper triangular matrix the determinant is just the product of diagonal entries.
        3) While transforming the original matrix, we keep track of how each row operation affects
           the determinant:
           - Row swap - flips the sign of determinant,
           - Multiply a row by a scalar - scales determinant by a scalar (this is usually avoided
             by dividing during elimination instead).
           - Add a multiple of one row to another - determinant unchanged.

        In short:
        - Repeatedly choose a pivot, eliminate below, swap if needed.
        - Multiply diagonal entries, correcting sign if swaps happened.
    */
    for (size_t i = 0; i < n; ++i) {
        // Partial pivoting: find max in column i
        size_t pivot = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(data[j][i]) > std::abs(data[pivot][i])) {
                pivot = j;
            }
        }

        if (std::abs(data[pivot][i]) < eps) {
            return 0.0;
        }

        if (pivot != i) {
            std::swap(data[pivot], data[i]);
            sign = -sign;
        }

        det *= data[i][i];

        // Eliminate below
        for (size_t j = i + 1; j < n; ++j) {
            double factor = data[j][i] / data[i][i];
            for (size_t k = i; k < n; ++k) {
                data[j][k] -= factor * data[i][k];
            }
        }
    }

    return det * sign;
}

Matrix Matrix::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Inverse requires a square matrix!");
    }
    size_t n = rows_;

    if (n == 0) {
        return Matrix(0, 0);
    }

    if (n == 1) {
        if (std::abs(data_[0][0]) < eps) {
            throw std::runtime_error("Inverse does not exist, because matrix is singular!");
        }
        return Matrix(1, 1, 1.0 / data_[0][0]);
    }

    // Augment matrix with identity
    vector<vector<double>> aug(n, vector<double>(2 * n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = data_[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    for (size_t i = 0; i < n; ++i) {
        // Find pivot row
        size_t pivot = i;
        for (size_t row = i + 1; row < n; ++row) {
            if (std::abs(aug[row][i]) > std::abs(aug[pivot][i])) {
                pivot = row;
            }
        }

        if (std::abs(aug[pivot][i]) < eps) {
            throw std::runtime_error("Inverse does not exist, because matrix is singular!");
        }

        if (pivot != i) {
            std::swap(aug[pivot], aug[i]);
        }

        // Normalize pivot row
        double diag = aug[i][i];
        for (size_t j = 0; j < 2 * n; ++j) {
            aug[i][j] /= diag;
        }

        // Eliminate other rows
        for (size_t row = 0; row < n; ++row) {
            if (row == i) {
                continue;
            }
            double factor = aug[row][i];
            for (size_t j = 0; j < 2 * n; ++j) {
                aug[row][j] -= factor * aug[i][j];
            }
        }
    }

    // Extract inverse from augmented matrix
    vector<vector<double>> inv_data(n, vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inv_data[i][j] = aug[i][n + j];
        }
    }

    return Matrix(std::move(inv_data));
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

Matrix Matrix::identity(size_t size) {
    Matrix mat(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        mat[i][i] = 1.0;
    }
    return mat;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}