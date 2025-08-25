#pragma once
#include <vector>
#include <ostream>

using std::vector;
using std::move;
using std::ostream;

class Matrix {
public:
    Matrix() = delete;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double val);
    Matrix(const Matrix& mat);
    Matrix(const vector<vector<double>>& data);
    Matrix(Matrix&& mat) noexcept;
    ~Matrix() = default;
    
    vector<double>& operator[](size_t row);
    const vector<double>& operator[](size_t row) const;

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    Matrix& operator=(const Matrix& mat);
    Matrix& operator=(Matrix&& mat) noexcept;
    Matrix& operator=(const vector<vector<double>>& data);

    bool operator==(const Matrix& mat) const;
    bool operator==(const vector<vector<double>>& data) const;
    
    bool operator!=(const Matrix& mat) const;
    bool operator!=(const vector<vector<double>>& data) const;

    Matrix operator+(double val) const;
    Matrix operator+(const Matrix& mat) const;
    
    Matrix operator-(double val) const;
    Matrix operator-(const Matrix& mat) const;

    Matrix operator*(double val) const;
    friend Matrix operator*(double val, const Matrix& mat);
    Matrix operator*(const Matrix& mat) const;

    Matrix operator-() const;

    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
    
    void fill(double val);
    Matrix transpose() const;

    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }
    bool is_empty() const { return data_.empty(); }
    bool equals(const Matrix& mat, double epsilon = 1e-9) const;
    bool equals(const vector<vector<double>>& data, double epsilon = 1e-9) const;
    friend ostream& operator<<(ostream&, const Matrix& matrix);

private:
    vector<vector<double>> data_;
    size_t rows_;
    size_t cols_;
};


/*
TODO:
- Prevent double overflow handling.
- Implement:
    - transpose()
    - assignment operators
    - identity()
    - resize()
    - determinant()
    - inverse()

*/