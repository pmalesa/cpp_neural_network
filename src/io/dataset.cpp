#include "dataset.h"
#include "spdlog/spdlog.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <set>

using std::ifstream;
using std::ofstream;
using std::getline;
using std::find_if;
using std::isspace;
using std::set;

// --------------------------------------------------
//  Constructors
// --------------------------------------------------

Dataset::Dataset()
    : path_(""), 
      size_(0), 
      target_column_(-1),
      headers_(false),
      index_column_(false),
      categorical_(false) {}

Dataset::Dataset(const string& path)
    : Dataset() {
    load_csv(path);
}

// --------------------------------------------------
//  I/O Operations
// --------------------------------------------------

void Dataset::load_csv(const string& path, bool headers, bool index_column, size_t target_column) {
    spdlog::info("Loading dataset from: '" + path + "'");
    
    path_ = path;
    headers_ = headers;
    index_column_ = index_column;
    target_column_ = target_column;
    size_ = 0;

    process_data_();
}

void Dataset::save_csv(const string& path) {
    spdlog::info("Saving dataset to: '" + path + "'");
    
    ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: '" + path + "'");
    }

    // Write headers if present
    if (headers_ && !header_names_.empty()) {
        for (size_t col = 0; col < header_names_.size(); ++col) {
            file << header_names_[col];
            if (col + 1 < header_names_.size()) {
                file << ',';
            }
        }
        file << '\n';
    }

    // Write data
    for (const auto& row : raw_data_) {
        for (size_t col = 0; col < row.size(); ++col) {
            file << row[col];
            if (col + 1 < row.size()) {
                file << ',';
            }
        }
        file << '\n';
    }
    file.close();
}

/* TODO */
void Dataset::print() const {
    
}

// --------------------------------------------------
//  Data Access
// --------------------------------------------------

Matrix Dataset::operator[](size_t row) const {
    if (row >= data_.get_rows()) {
        throw std::out_of_range("Row index out of bounds.");
    }
    return data_[row];
}

Matrix Dataset::get_range(size_t start_row, size_t end_row) const {
    if (start_row >= end_row || end_row > data_.get_rows()) {
        throw std::invalid_argument("Invalid dataset range: [" +
            std::to_string(start_row) + ", " + std::to_string(end_row) + ")");
    }

    size_t n_rows = end_row - start_row;
    Matrix result(n_rows, data_.get_cols());
    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t col = 0; col < data_.get_cols(); ++col) {
            result[row][col] = data_[start_row + row][col];
        }
    }
    return result;
}

const vector<string>& Dataset::get_row(size_t row) const {
    if (row >= raw_data_.size()) {
        throw std::out_of_range("Row index out of bounds.");
    }
    return raw_data_[row];
}

// --------------------------------------------------
//  Configuration setters
// --------------------------------------------------

void Dataset::set_target_column(size_t target_column) {
    if (target_column_ == static_cast<long long>(target_column)) {
        return;
    }
    target_column_ = static_cast<long long>(target_column);
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

// --------------------------------------------------
//  CSV Parsing & Conversion
// --------------------------------------------------

void Dataset::process_data_() {
    ifstream file(path_);
    if (!file.is_open()) {
        string message = "Failed to open file: " + path_;
        spdlog::error(message);
        throw std::runtime_error(message);
    }
    
    // Erase any existing data
    header_names_.clear();
    raw_data_.clear();

    // Obtain raw data
    string line;
    while (getline(file, line)) {
        auto tokens = split_csv_line_(line);
        if (headers_ && header_names_.empty()) {
            header_names_ = std::move(tokens);
            continue;
        }
        raw_data_.push_back(tokens);
    }
    if (raw_data_.empty()) {
        string message = "Empty dataset file: '" + path_ + "'";
        spdlog::error(message);
        throw std::runtime_error(message);
    }

    // Verify number of columns
    size_t n_columns = raw_data_[0].size();
    for (auto& row : raw_data_) {
        if (row.size() != n_columns) {
            string message = "Loaded CSV file is corrupted - different number of columns in rows.";
            spdlog::error(message);
            throw std::invalid_argument(message);
        }
    }

    // Default to last column if invalid
    if (target_column_ < 0 || target_column_ > static_cast<long long>(raw_data_[0].size()) - 1) {
        target_column_ = static_cast<long long>(raw_data_[0].size()) - 1;
    }
    if (target_column_ < 0) {
        string message = "Target column could not be set correctly - dataset empty.";
        spdlog::error(message);
        throw std::invalid_argument(message);
    }

    spdlog::info("Raw CSV processed successfully.");
    convert_to_numerical_();
}

void Dataset::convert_to_numerical_() {
    if (is_data_categorical_()) {
        string message = "Categorical data procesing not supported!";
        spdlog::error(message);
        throw std::invalid_argument(message);
    }

    const size_t n_cols = raw_data_[0].size();
    const size_t n_rows = raw_data_.size();
    const size_t target_col = static_cast<size_t>(target_column_);
    const size_t n_features = n_cols - (index_column_ ? 2 : 1);
    size_ = n_rows;

    /* Convert raw data into numerical */
    data_ = Matrix(size_, n_features);
    size_t feature_col = 0;
    for (size_t col = 0; col < n_cols; ++col) {
        if ((index_column_ && col == 0) || col == target_col) {
            continue;
        }
        for (size_t row = 0; row < n_rows; ++row) {
            data_[row][feature_col] = std::stod(raw_data_[row][col]);
        }
        ++feature_col;
    }

    label_to_int_map_.clear();
    int_to_label_map_.clear();

    /* Convert the target column to mapped integer values (from N to N-1) */
    targets_ = Matrix(size_, 1);
    if (is_target_column_categorical_()) {
        set<string> unique_labels;
        for (const auto& row : raw_data_) {
            unique_labels.insert(row[target_col]);
        }
        int id = 0;
        for (const auto& label : unique_labels) {
            label_to_int_map_[label] = id;
            int_to_label_map_[id] = label;
            ++id;
        }
        for (size_t row = 0; row < n_rows; ++row) {
            targets_[row][0] = static_cast<double>(label_to_int_map_.at(raw_data_[row][target_col]));
        }
    } else {
        for (size_t row = 0; row < n_rows; ++row) {
            targets_[row][0] = std::stod(raw_data_[row][target_col]);
        }
    }

    spdlog::info("Conversion to numerical data successful.");
}

vector<string> Dataset::split_csv_line_(const string& line) const {
    vector<string> result;
    string cell;
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            trim_(cell);
            result.push_back(std::move(cell));
            cell.clear();
        } else {
            cell += ch;
        }
    }

    trim_(cell);
    result.push_back(std::move(cell));
    return result;
}

void Dataset::trim_(string& str) const {
    str.erase(str.begin(), find_if(str.begin(), str.end(), 
        [](unsigned char ch) { return !isspace(ch); }));
    str.erase(find_if(str.rbegin(), str.rend(),
        [](unsigned char ch) { return !isspace(ch); }).base(),
        str.end());
}

/**
 * It checks whether the loaded raw_data_ contains
 * any categorical data (excluding the target column).
*/
bool Dataset::is_data_categorical_() const {
    size_t target_col = static_cast<size_t>(target_column_);
    for (const auto& row : raw_data_) {
        for (size_t col = 0; col < row.size(); ++col) {
            if (col == target_col) {
                continue;
            }
            if (!is_float_(row[col])) {
                return true;
            }
        }

    }
    return false;
}

/**
 * It checks whether the given string is
 * a floating point number.
*/
bool Dataset::is_float_(const string& float_str) const {
    try {
        size_t pos;
        std::stod(float_str, &pos);
        return pos == float_str.size();
    } catch (...) {
        return false;
    }
}

bool Dataset::is_target_column_categorical_() const {
    size_t target_col = static_cast<size_t>(target_column_);
    for (const auto& row : raw_data_) {
        if (!is_float_(row[target_col])) {
            return true;
        }
    }
    return false;
}
