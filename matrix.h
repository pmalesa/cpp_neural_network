#pragma once
#include <vector>

using std::vector;
using std::move;

class Matrix {
public:
    Matrix() = delete;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double val);
    Matrix(const Matrix& mat);
    Matrix(Matrix&& mat) noexcept;
    ~Matrix() = default;
    
    vector<double>& operator[](size_t row);
    const vector<double>& operator[](size_t row) const;

    Matrix& operator=(const Matrix& mat);
    Matrix& operator=(Matrix&& mat) noexcept;

    Matrix operator*(double val);
    Matrix operator*(const Matrix& mat);

    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
    
    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }
    bool is_empty() const { return data_.empty(); }

private:
    vector<vector<double>> data_;
    size_t rows_;
    size_t cols_;
};