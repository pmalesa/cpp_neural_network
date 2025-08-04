#pragma once
#include <vector>

using std::vector;
using std::move;

class Matrix {
public:
    Matrix() = delete;
    Matrix(size_t rows, size_t cols);
    Matrix(const Matrix& mat);
    Matrix(Matrix&& mat) noexcept;
    ~Matrix();

    // TODO - implement operator[]. 
    Matrix& operator=(const Matrix& mat);
    Matrix& operator=(Matrix&& mat) noexcept;

    double at(size_t row, size_t col) const;
    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }

private:
    vector<vector<double>> data_;
    size_t rows_;
    size_t cols_;
};