#pragma once
#include <vector>
#include <ostream>

using std::vector;
using std::move;
using std::ostream;
using std::initializer_list;

class Matrix {
public:
    Matrix() = default;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double val);
    Matrix(const Matrix& mat);
    Matrix(const vector<double>& data); // Create 1 x N matrix (row vector)
    Matrix(const vector<vector<double>>& data);
    Matrix(Matrix&& mat) noexcept;
    Matrix(vector<vector<double>>&& data);
    Matrix(initializer_list<initializer_list<double>> init_list);
    ~Matrix() = default;
    
    vector<double>& operator[](size_t row);
    const vector<double>& operator[](size_t row) const;

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    Matrix& operator=(const Matrix& mat);
    Matrix& operator=(Matrix&& mat) noexcept;
    Matrix& operator=(const vector<vector<double>>& data);
    Matrix& operator=(vector<vector<double>>&& data);
    Matrix& operator=(initializer_list<initializer_list<double>> init_list);

    bool operator==(const Matrix& mat) const;
    bool operator==(const vector<vector<double>>& data) const;
    bool operator==(initializer_list<initializer_list<double>> init_list) const;
    friend bool operator==(initializer_list<initializer_list<double>> init_list, const Matrix& mat) { return mat == init_list; }
    
    bool operator!=(const Matrix& mat) const;
    bool operator!=(const vector<vector<double>>& data) const;
    bool operator!=(initializer_list<initializer_list<double>> init_list) const;
    friend bool operator!=(initializer_list<initializer_list<double>> init_list, const Matrix& mat) { return mat != init_list; }

    Matrix operator+(double val) const;
    friend Matrix operator+(double val, const Matrix& mat);
    Matrix operator+(const Matrix& mat) const;
    
    Matrix operator-(double val) const;
    Matrix operator-(const Matrix& mat) const;

    Matrix operator*(double val) const;
    friend Matrix operator*(double val, const Matrix& mat);
    Matrix operator*(const Matrix& mat) const;
    Matrix operator*(const vector<double>& vec) const;
    friend Matrix operator*(const vector<double>& vec, const Matrix& mat);

    Matrix operator-() const;

    Matrix elementwise_mul(const Matrix& mat) const;

    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
    
    void fill(double val);
    Matrix transpose() const;
    void resize(size_t new_rows, size_t new_cols);
    void reshape(size_t new_rows, size_t new_cols);
    double det() const;
    double trace() const; // Sum of diagonal elements
    double min() const;
    double max() const;
    double sum() const;
    Matrix inverse() const;
    Matrix flatten() const;
    void fill_random(double min = -1.0, double max = 1.0);

    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }
    bool is_empty() const { return rows_ == 0 || cols_ == 0; }
    bool equals(const Matrix& mat) const;
    bool equals(const vector<vector<double>>& data) const;
    bool equals(initializer_list<initializer_list<double>> init_list) const;
    friend ostream& operator<<(ostream&, const Matrix& matrix);

    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix diagonal(size_t size, double val);
    
private:
    static constexpr double eps = 1e-9;
    static constexpr size_t PARALLEL_THRESHOLD = 256;

    vector<vector<double>> data_;
    size_t rows_;
    size_t cols_;

    void validate_and_set_shape_(const vector<vector<double>>& vec);
    void assign_from_(const vector<vector<double>>& vec);
    void assign_from_(const vector<vector<double>>&& vec);
    void assign_from_(initializer_list<initializer_list<double>> init_list);

    void add_sequentially_(double val, Matrix& result) const;
    void add_sequentially_(const Matrix& mat, Matrix& result) const;
    
    void add_concurrently_(double val, Matrix& result) const;
    void add_concurrently_(const Matrix& mat, Matrix& result) const;

    void multiply_sequentially_(double val, Matrix& result) const;
    void multiply_sequentially_(const Matrix& mat, Matrix& result) const;
    void multiply_concurrently_(double val, Matrix& result) const;
    void multiply_concurrently_(const Matrix& mat, Matrix& result) const;
};

/*
TODO:
    - write class to benchmark time computations, and compare concurrent vs sequential on big matrices
    - sum()
    - mean()
*/