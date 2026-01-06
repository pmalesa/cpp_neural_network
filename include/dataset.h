#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"

#include <string>
#include <vector>
#include <unordered_map>

using std::string;
using std::vector;
using std::unordered_map;

/**
 * Simple class for loading and saving data in CSV format.
 * 
 * [WARNING]
 * Categorical data is not supported yet!
 * This does not include column with target values.
*/
class Dataset {
public:
    Dataset();
    Dataset(const string& path);
    ~Dataset() = default;

    void load_csv(
        const string& path, 
        bool headers = false, 
        bool index_column = false,
        size_t target_column = -1
    );  // throws std::runtime_error and std::invalid_argument
    void save_csv(const string& path);

    Matrix operator[](size_t row) const;
    Matrix get_data() const { return data_; }
    Matrix get_targets() const { return targets_; }
    const vector<vector<string>>& get_raw_data() const { return raw_data_; }
    const vector<string>& get_raw_targets() const { return raw_data_[target_column_]; }
    Matrix get_range(size_t start, size_t end) const;
    const vector<string>& get_row(size_t row) const;
    const vector<string>& get_headers() const { return header_names_; }
    size_t size() const { return size_; }
    size_t get_columns_count() const { return header_names_.size(); }
    size_t get_features_count() const { return data_.get_cols(); }
    bool empty() const { return size_ == 0; }
    void print() const;
    void set_target_column(size_t target_column);   // throws std::runtime_error and std::invalid_argument
    void set_index_column(bool index_column);       // throws std::runtime_error and std::invalid_argument
    void set_headers(bool headers);                 // throws std::runtime_error and std::invalid_argument

private:
    void process_data_();
    void convert_to_numerical_();
    vector<string> split_csv_line_(const string& line) const;
    void trim_(string& str) const;
    bool is_data_categorical_() const;
    bool is_float_(const string& float_str) const;
    bool is_target_column_categorical_() const;

    string path_;
    vector<vector<string>> raw_data_;
    Matrix data_;       // Contains all numerical data (without target values)
    Matrix targets_;    // Contains targets' numerical data
    vector<string> header_names_;
    unordered_map<string, int> label_to_int_map_;
    unordered_map<int, string> int_to_label_map_;
    size_t size_;
    long long target_column_;
    bool headers_;
    bool index_column_;
    bool categorical_;  // Does not concern the target column
};

#endif // DATASET_H