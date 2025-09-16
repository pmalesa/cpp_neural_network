#include "matrix.h"
#include <stdexcept>
#include <thread>
#include <cmath>
#include <future>
#include <random>
#include <algorithm>

using std::thread;

Matrix::Matrix(size_t rows, size_t cols)
    : data_(vector<vector<double>>(rows, vector<double>(cols, 0.0))), rows_(rows), cols_(cols) {}

Matrix::Matrix(size_t rows, size_t cols, double val) 
    : data_(vector<vector<double>>(rows, vector<double>(cols, val))), rows_(rows), cols_(cols) {}

Matrix::Matrix(const Matrix& mat)
    : data_(mat.data_), rows_(mat.rows_), cols_(mat.cols_) {} 

Matrix::Matrix(const vector<vector<double>>& data) {
    assign_from_(data);
}

Matrix::Matrix(Matrix&& mat) noexcept
    : data_(move(mat.data_)), rows_(mat.rows_), cols_(mat.cols_) {
    mat.rows_ = 0;
    mat.cols_ = 0;
}

Matrix::Matrix(vector<vector<double>>&& data) {
    assign_from_(std::move(data));
}

Matrix::Matrix(initializer_list<initializer_list<double>> init_list) {
    assign_from_(init_list);
}

vector<double>& Matrix::operator[](size_t row) {
    return data_[row];
}

const vector<double>& Matrix::operator[](size_t row) const {
    return data_[row];
}

double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows_ || row < 0 || col >= cols_ || col < 0) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return data_[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return data_[row][col];
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
    assign_from_(data);
    return *this;
}

Matrix& Matrix::operator=(vector<vector<double>>&& data) {
    assign_from_(std::move(data));
    return *this;
}

Matrix& Matrix::operator=(initializer_list<initializer_list<double>> init_list) {
    assign_from_(init_list);
    return *this;
}

bool Matrix::operator==(const Matrix& mat) const {
    return equals(mat);
}

bool Matrix::operator==(const vector<vector<double>>& data) const {
    return equals(data);
}

bool Matrix::operator==(initializer_list<initializer_list<double>> init_list) const {
    return equals(init_list);
}

bool Matrix::operator!=(const Matrix& mat) const {
    return !(*this == mat);
}

bool Matrix::operator!=(const vector<vector<double>>& data) const {
    return !(*this == data);
}

bool Matrix::operator!=(initializer_list<initializer_list<double>> init_list) const {
    return !(*this == init_list);
}

Matrix Matrix::operator+(double val) const {
    Matrix result(rows_, cols_);
    if (rows_ < PARALLEL_THRESHOLD) {
        add_sequentially_(val, result);
    } else {
        add_concurrently_(val, result);
    }
    return result;
}

Matrix operator+(double val, const Matrix& mat) {
    return mat + val;
}

Matrix Matrix::operator+(const Matrix& mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::domain_error("Matrices can not be added!");
    }
    Matrix result(rows_, cols_);
    if (rows_ < PARALLEL_THRESHOLD) {
        add_sequentially_(mat, result);
    } else {
        add_concurrently_(mat, result);
    }
    return result;
}

Matrix Matrix::operator-(double val) const {
    return (*this) + (-val);
}

Matrix Matrix::operator-(const Matrix& mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::domain_error("Matrices can not be subtracted!");
    }
    return (*this) + (-mat);
}

Matrix Matrix::operator*(double val) const {
    Matrix result(rows_, cols_);
    if (rows_ < PARALLEL_THRESHOLD) {
        multiply_sequentially_(val, result);
    } else {
        multiply_concurrently_(val, result);
    }
    return result;
}

Matrix operator*(double val, const Matrix& mat) {
    return mat * val;
}

Matrix Matrix::operator*(const Matrix& mat) const {
    if (cols_ != mat.rows_) {
        throw std::domain_error("Matrices can not be multiplied!");
    }
    Matrix result(rows_, mat.cols_);
    if (rows_ * mat.cols_ < PARALLEL_THRESHOLD * PARALLEL_THRESHOLD) {
        multiply_sequentially_(mat, result);
    } else {
        multiply_concurrently_(mat, result);
    }
    return result;
}

Matrix Matrix::operator-() const {
    Matrix result(rows_, cols_);
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            result[row][col] = -data_[row][col];
        }
    }
    return result;
}

Matrix Matrix::elementwise_mul(const Matrix& mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::domain_error("Matrices can not be multiplied!");
    }
    Matrix result(rows_, cols_);
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            long double v_ld = static_cast<long double>(data_[row][col]) *
                               static_cast<long double>(mat[row][col]);
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Multiplication overflowed!");
            }
            result[row][col] = v_d;
        }
    }    
    return result;
}

double& Matrix::at(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return data_[row][col];
}

const double& Matrix::at(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds!");
    }
    return data_[row][col];
}

void Matrix::fill(double val) {
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            data_[row][col] = val;
        }
    }
}

Matrix Matrix::transpose() const {
    Matrix mat(cols_, rows_);
    for (size_t i = 0; i < cols_; ++i) {
        for (size_t j = 0; j < rows_; ++j) {
            mat[i][j] = data_[j][i];
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

void Matrix::reshape(size_t new_rows, size_t new_cols) {
    if (rows_ * cols_ != new_rows * new_cols) {
        throw std::invalid_argument("Total number of elements after the reshape operation must be preserved!");
    }
    Matrix result(new_rows, new_cols);
    size_t idx = 0;
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            size_t new_row = idx / new_cols;
            size_t new_col = idx % new_cols;
            result[new_row][new_col] = data_[row][col];
            ++idx;
        }
    }
    data_ = std::move(result.data_);
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

double Matrix::trace() const {
    if (rows_ != cols_ || rows_ == 0) {
        throw std::domain_error("Trace requires a non-empty square matrix!");
    }
    long double v_ld = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        v_ld += static_cast<long double>(data_[i][i]);
    }
    double result = static_cast<double>(v_ld);
    if (!std::isfinite(result)) {
        throw std::overflow_error("Addition/subtraction overflowed!");
    }
    return result;
}

double Matrix::min() const {
    if (rows_ == 0 || cols_ == 0) {
        throw std::domain_error("Matrix can not be empty!");
    }
    double min = data_[0][0];
    for (auto& row : data_) {
        double row_min = *std::min_element(row.begin(), row.end());
        min = std::min(min, row_min);
    }
    return min;
}

double Matrix::max() const {
    if (rows_ == 0 || cols_ == 0) {
        throw std::domain_error("Matrix can not be empty!");
    }
    double max = data_[0][0];
    for (auto& row : data_) {
        double row_max = *std::max_element(row.begin(), row.end());
        max = std::max(max, row_max);
    }
    return max;
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

Matrix Matrix::flatten() const {
    if (rows_ == 0 || cols_ == 0) {
        return {};
    }

    if (rows_ == 1) {
        return *this;
    }

    Matrix result(1, rows_ * cols_);
    size_t i = 0;
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            result[0][i] = data_[row][col];
            ++i;
        }
    }

    return result;
}

void Matrix::fill_random(double min, double max) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            data_[row][col] = dist(gen);
        }
    }
}

bool Matrix::equals(const Matrix& mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        return false;
    }
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            if (abs(data_[row][col] - mat.data_[row][col]) > eps) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::equals(const vector<vector<double>>& data) const {
    if (data.empty()) {
        return data_.empty();
    }
    if (rows_ != data.size() || cols_ != data[0].size()) {
        return false;
    }
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            if (abs(data_[row][col] - data[row][col]) > eps) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::equals(initializer_list<initializer_list<double>> init_list) const {
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

Matrix Matrix::diagonal(size_t size, double val) {
    Matrix mat(size, size);
    for (size_t i = 0; i < mat.rows_; ++i) {
        mat[i][i] = val;
    }
    return mat;
}

void Matrix::validate_and_set_shape_(const vector<vector<double>>& vec) {
    if (vec.empty()) {
        rows_ = 0;
        cols_ = 0;
        return;
    }
    cols_ = vec[0].size();
    for (const auto& row : vec) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns!");
        }
    }
    rows_ = vec.size();
}

void Matrix::assign_from_(const vector<vector<double>>& vec) {
    validate_and_set_shape_(vec);
    data_ = vec;
}

void Matrix::assign_from_(const vector<vector<double>>&& vec) {
    validate_and_set_shape_(vec);
    data_ = std::move(vec);
}

void Matrix::assign_from_(initializer_list<initializer_list<double>> init_list) {
    rows_ = init_list.size();
    cols_ = rows_ ? init_list.begin()->size() : 0;
    for (const auto& row : init_list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns!");
        }
    }
    data_.clear();
    data_.reserve(rows_);
    for (const auto& row : init_list) {
        data_.emplace_back(row);
    }
}

void Matrix::add_sequentially_(double val, Matrix& result) const {
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            long double v_ld = static_cast<long double>(data_[row][col]) +
                               static_cast<long double>(val);
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Addition/subtraction overflowed!");
            }
            result[row][col] = v_d;
        }
    }
}

void Matrix::add_sequentially_(const Matrix& mat, Matrix& result) const {
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t col = 0; col < cols_; ++col) {
            long double v_ld = static_cast<long double>(data_[row][col]) + 
                               static_cast<long double>(mat.data_[row][col]);
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Addition/subtraction overflowed!");
            }
            result[row][col] = v_d;
        }
    }
}
    
void Matrix::add_concurrently_(double val, Matrix& result) const {
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2;
    }

    // formula for ceil() math function on integers, adding the (divisor - 1) bumps the remainder fraction to 1
    size_t chunk = (rows_ + num_threads - 1) / num_threads; 

    // Define lambda function to calculate cell value (division of calculations on rows)
    auto worker = [&](size_t start, size_t end) {
        for (size_t row = 0; row < rows_; ++row) {
            for (size_t col = 0; col < cols_; ++col) {
                long double v_ld = static_cast<long double>(data_[row][col]) +
                                   static_cast<long double>(val);
                double v_d = static_cast<double>(v_ld);
                if (!std::isfinite(v_d)) {
                    throw std::overflow_error("Addition/subtraction overflowed!");
                }
                result[row][col] = v_d;
            }
        }
    };

    // Divide the calculations among threads
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = start + chunk;
        futures.push_back(std::async(std::launch::async, worker, start, end));
    }
    
    // Propagate exceptions to the main thread, if any were thrown
    for (auto& f : futures) {
        f.get();
    }
}

void Matrix::add_concurrently_(const Matrix& mat, Matrix& result) const {
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2;
    }

    // formula for ceil() math function on integers, adding the (divisor - 1) bumps the remainder fraction to 1
    size_t chunk = (rows_ + num_threads - 1) / num_threads; 

    // Define lambda function to calculate cell value (division of calculations on rows)
    auto worker = [&](size_t start, size_t end) {
        for (size_t row = 0; row < rows_; ++row) {
            for (size_t col = 0; col < cols_; ++col) {
                long double v_ld = static_cast<long double>(data_[row][col]) + 
                                   static_cast<long double>(mat.data_[row][col]);
                double v_d = static_cast<double>(v_ld);
                if (!std::isfinite(v_d)) {
                    throw std::overflow_error("Addition/subtraction overflowed!");
                }
                result[row][col] = v_d;
            }
        }
    };

    // Divide the calculations among threads
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = start + chunk;
        futures.push_back(std::async(std::launch::async, worker, start, end));
    }
    
    // Propagate exceptions to the main thread, if any were thrown
    for (auto& f : futures) {
        f.get();
    }
}

void Matrix::multiply_sequentially_(double val, Matrix& result) const {
    for (size_t row = 0; row < rows_; ++row) {
        long double v_ld = 0.0;
        for (size_t col = 0; col < cols_; ++col) {
            v_ld = static_cast<long double>(data_[row][col]) * 
                   static_cast<long double>(val);
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Matrix multiplication overflowed!");
            }
            result[row][col] = v_d;
        }
    }
}

void Matrix::multiply_sequentially_(const Matrix& mat, Matrix& result) const {
    size_t common_dim = cols_; 
    for (size_t row = 0; row < result.rows_; ++row) {
        for (size_t col = 0; col < result.cols_; ++col) {
            long double v_ld = 0.0;
            for (size_t i = 0; i < common_dim; ++i) {
                v_ld += static_cast<long double>(data_[row][i]) * 
                       static_cast<long double>(mat.data_[i][col]);
            }
            double v_d = static_cast<double>(v_ld);
            if (!std::isfinite(v_d)) {
                throw std::overflow_error("Matrix multiplication overflowed!");
            }
            result[row][col] = v_d;
        }
    }
}

void Matrix::multiply_concurrently_(double val, Matrix& result) const {
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2;
    }

    // formula for ceil() math function on integers, adding the (divisor - 1) bumps the remainder fraction to 1
    size_t chunk = (rows_ + num_threads - 1) / num_threads; 

    // Define lambda function to calculate cell value (division of calculations on rows)
    auto worker = [&](size_t start, size_t end) {
        for (size_t row = start; row < end && row < rows_; ++row) {
            long double v_ld = 0.0;
            for (size_t col = 0; col < cols_; ++col) {
                v_ld = static_cast<long double>(data_[row][col]) * 
                    static_cast<long double>(val);
                double v_d = static_cast<double>(v_ld);
                if (!std::isfinite(v_d)) {
                    throw std::overflow_error("Matrix multiplication overflowed!");
                }
                result[row][col] = v_d;
            }
        }
    };

    // Divide the calculations among threads
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = start + chunk;
        futures.push_back(std::async(std::launch::async, worker, start, end));
    }
    
    // Propagate exceptions to the main thread, if any were thrown
    for (auto& f : futures) {
        f.get();
    }
}

void Matrix::multiply_concurrently_(const Matrix& mat, Matrix& result) const {
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2;
    }
   
    // formula for ceil() math function on integers, adding the (divisor - 1) bumps the remainder fraction to 1
    size_t chunk = (rows_ + num_threads - 1) / num_threads;

    // Define lambda function to calculate cell value (division of calculations on rows)
    auto worker = [&](size_t start, size_t end) {
        size_t common_dim = cols_; 
        for (size_t row = start; row < end && row < rows_; ++row) {
            for (size_t col = 0; col < mat.cols_; ++col) {
                long double sum = 0;
                for (size_t i = 0; i < common_dim; ++i) {
                    sum += static_cast<long double>(data_[row][i]) *
                           static_cast<long double>(mat.data_[i][col]);
                }
                double v = static_cast<double>(sum);
                if (!std::isfinite(v)) {
                    throw std::overflow_error("Matrix multiplication overflowed!");
                }
                result[row][col] = v;
            }
        }
    };

    // Divide the calculations among threads
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = start + chunk;
        futures.push_back(std::async(std::launch::async, worker, start, end));
    }
    
    // Propagate exceptions to the main thread, if any were thrown
    for (auto& f : futures) {
        f.get();
    }
}