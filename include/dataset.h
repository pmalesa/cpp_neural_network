#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"

#include <string>
#include <vector>
#include <unordered_map>

using std::string;
using std::vector;
using std::unordered_map;

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
    );
    void save_csv(const string& path);

    Matrix operator[](size_t row) const;
    Matrix get_data() const { return data_; }
    Matrix get_targets() const { return data_[data_.get_cols() - 1]; }
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
    void set_target_column(size_t target_column);
    void set_index_column(bool index_column);
    void set_headers(bool headers);

private:
    void process_data_();
    void convert_to_numerical_();
    vector<string> split_csv_line_(const string& line) const;
    void trim_(string& str) const;

    string path_;
    vector<vector<string>> raw_data_;
    Matrix data_;
    vector<string> header_names_;
    unordered_map<string, int> label_to_int_map_;
    unordered_map<string, int> int_to_label_map_;
    size_t size_;
    size_t target_column_;
    bool headers_;
    bool index_column_;
    bool categorical_;  // Does not concern the target column
};

#endif // DATASET_H