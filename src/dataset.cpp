#include "dataset.h"
#include "logger.h"

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

static Logger& logger = Logger::instance();

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
    logger.log("Loading dataset from: '" + path + "'", Logger::Level::Info);
    
    path_ = path;
    headers_ = headers;
    index_column_ = index_column;
    target_column_ = target_column;
    size_ = 0;

    process_data_();
}

void Dataset::save_csv(const string& path) {
    logger.log("Saving dataset to: '" + path + "'", Logger::Level::Info);
    
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
//  CSV Parsing & Conversion (TODO - REFACTOR)
// --------------------------------------------------

void Dataset::process_data_() {
    ifstream file(path_);
    if (!file.is_open()) {
        string message = "Failed to open file: " + path_;
        logger.log(message, Logger::Level::Error);
        throw std::runtime_error(message);
    }
    
    // Erase any existing data
    header_names_.clear();
    raw_data_.clear();

    // Obtain raw data
    string line;
    while (getline(file, line)) {
        vector<string> line_splitted = split_csv_line_(line);
        if (headers_ && header_names_.empty()) {
            header_names_ = line_splitted;
            continue;
        }
        raw_data_.push_back(line_splitted);
    }
    if (raw_data_.empty()) {
        return;
    }
    if (target_column_ < 0 || target_column_ > static_cast<long long>(raw_data_[0].size()) - 1) {
        target_column_ = static_cast<long long>(raw_data_[0].size()) - 1;
    }
    if (target_column_ < 0) {
        string message = "[ERROR] Target column could not be processed correctly.";
        logger.log(message, Logger::Level::Error);
        throw std::invalid_argument(message);
    }

    // Verify number of columns
    size_t n_columns = raw_data_[0].size();
    for (auto& row : raw_data_) {
        if (row.size() != n_columns) {
            string message = "Loaded CSV file is corrupted - different number of columns in rows.";
            logger.log(message, Logger::Level::Error);
            throw std::invalid_argument(message);
        }
    }
    logger.log("Raw data processed successfully.", Logger::Level::Info);

    convert_to_numerical_();
}

void Dataset::convert_to_numerical_() {
    if (is_data_categorical_()) {
        string message = "Categorical data procesing not supported!";
        logger.log(message, Logger::Level::Error);
        throw std::invalid_argument(message);
    }

    /* Convert raw data into numerical */
    size_t n_features = raw_data_[0].size() - 1; 
    if (index_column_) {
        --n_features;
    }
    size_ = raw_data_.size();
    size_t current_column = 0;
    size_t target_column = static_cast<size_t>(target_column_);
    data_ = Matrix(size_, n_features);
    for (size_t col = 0; col < raw_data_[0].size(); ++col) {
        if ((index_column_ && col == 0) || col == target_column) {
            continue;
        }
        for (size_t row = 0; row < raw_data_.size(); ++row) {
            data_[row][current_column] = std::stod(raw_data_[row][col]);
        }
        ++current_column;
    }

    /* Convert the target column to mapped integer values (from N to N-1) */
    targets_ = Matrix(size_, 1);
    label_to_int_map_.clear();
    int_to_label_map_.clear();
    if (is_target_column_categorical_()) {
        set<string> unique_labels;
        for (size_t row = 0; row < raw_data_.size(); ++row) {
            unique_labels.insert(raw_data_[row][target_column]);
        }
        int label = 0;
        for (auto it = unique_labels.begin(); it != unique_labels.end(); ++it) {
            label_to_int_map_.insert({*it, label});
            int_to_label_map_.insert({label, *it});
            ++label;
        }
        for (size_t row = 0; row < size_; ++row) {
            targets_[row][0] = static_cast<double>(label_to_int_map_[raw_data_[row][target_column]]);
        }
    } else {
        for (size_t row = 0; row < size_; ++row) {
            targets_[row][0] = std::stod(raw_data_[row][target_column]);
        }
    }

    logger.log("Conversion to numerical data successful.", Logger::Level::Info);
}

vector<string> Dataset::split_csv_line_(const string& line) const {
    vector<string> result;
    string cell;
    bool in_quotes;

    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            trim_(cell);
            result.push_back(cell);
            cell.clear();
        } else {
            cell += ch;
        }
    }

    trim_(cell);
    result.push_back(cell);
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
    size_t target_column = static_cast<size_t>(target_column_);
    for (size_t row = 0; row < raw_data_.size(); ++row) {
        for (size_t col = 0; col < raw_data_[0].size(); ++col) {
            if (col == target_column) {
                continue;
            }
            if (!is_float_(raw_data_[row][col])) {
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
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

bool Dataset::is_target_column_categorical_() const {
    size_t target_column = static_cast<size_t>(target_column_);
    for (size_t row = 0; row < raw_data_.size(); ++row) {
        if (!is_float_(raw_data_[row][target_column])) {
            return true;
        }
    }
    return false;
}
