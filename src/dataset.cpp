#include "dataset.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using std::ifstream;
using std::getline;
using std::stringstream;

Dataset::Dataset()
    : path_(""), size_(0), target_column_(-1), headers_(false), index_column_(false), categorical_(false) { }

Dataset::Dataset(const string& path)
    : path_(path), size_(0), target_column_(-1), headers_(false), index_column_(false), categorical_(false) {
    load_csv(path);
}

void Dataset::load_csv(const string& path, size_t target_column, bool headers, bool index_column) {
    headers_ = headers;
    index_column_ = index_column;
    process_data_();
    if (!raw_data_.empty() && (target_column < 0 || target_column > raw_data_[0].size() - 1)) {
        target_column_ = raw_data_[0].size() - 1;
    }
}

/* TODO */
void Dataset::save_csv(const string& path) {
    
}

Matrix Dataset::operator[](size_t row) const {
    return data_[row];
}

/* TODO */
Matrix Dataset::get_range(size_t start, size_t end) const {
    // Return a range from start to end as Matrix object (inlcusive)

    return data_[end];
}


const vector<string>& Dataset::get_row(size_t row) const {
    return raw_data_[row];
}

/* TODO */
void Dataset::print() const {
    
}

void Dataset::set_target_column(size_t target_column) {
    if (target_column_ == target_column) {
        return;
    }
    target_column_ = target_column;
    process_data_();
}

void Dataset::set_index_column(bool index_column) {
    if (index_column_ == index_column) {
        return;
    }
    index_column_ = index_column;
    process_data_();
}

void Dataset::set_headers(bool headers) {
    if (headers_ == headers) {
        return;
    }
    headers_ = headers;
    process_data_();
}

/* TODO */
void Dataset::process_data_() {
    size_t first_row = headers_ ? 1 : 0;
    size_t first_col = index_column_ ? 1 : 0;

    ifstream file(path_);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path_);
    }

    string line;
    while (getline(file, line)) {

    }

    convert_to_numerical_();
}

/* TODO */
void Dataset::convert_to_numerical_() {
    if (raw_data_.empty()) {
        return;
    }

}

/* TODO */
vector<string> Dataset::split_csv_line(const string& line) const {
    return {};
}