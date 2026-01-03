#include "dataset.h"
#include "logger.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

using std::ifstream;
using std::getline;
using std::find_if;
using std::isspace;

static Logger& logger = Logger::instance();

Dataset::Dataset()
    : path_(""), size_(0), target_column_(-1), headers_(false), index_column_(false), categorical_(false) { }

Dataset::Dataset(const string& path)
    : path_(path), size_(0), target_column_(-1), headers_(false), index_column_(false), categorical_(false) {
    load_csv(path);
}

void Dataset::load_csv(const string& path, bool headers, bool index_column, size_t target_column) {
    path_ = path;
    headers_ = headers;
    index_column_ = index_column;
    target_column_ = target_column;
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

void Dataset::process_data_() {
    ifstream file(path_);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path_);
    }
    
    string line;
    header_names_.clear();
    while (getline(file, line)) {
        vector<string> line_splitted = split_csv_line_(line);
        if (headers_ && header_names_.empty()) {
            header_names_ = line_splitted;
            continue;
        }
        raw_data_.push_back(line_splitted);
    }
    logger.log("Raw data processed successfully.", Logger::Level::Info);
    convert_to_numerical_();
    logger.log("Conversion to numerical data successful.", Logger::Level::Info);
}

/* TODO */
void Dataset::convert_to_numerical_() {
    if (raw_data_.empty()) {
        return;
    }

    if (is_data_categorical_()) {
        logger.log("Categorical data found.", Logger::Level::Info);
        /*
            1. Calculate the number of categorical columns (except  target column)
            2. Determine the number of columns for each categorical columns
            3. Initialize the data_ Matrix object with the correct number of rows and columns.
        */
    } else {
        logger.log("No categorical data found.", Logger::Level::Info);    
    }

    /* Convert raw data into numerical */


    /* Convert the target column to mapped integer values (from N to N-1) */


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
    for (size_t row = 0; row < raw_data_.size(); ++row) {
        for (size_t col = 0; col < raw_data_[0].size(); ++col) {
            if (col == target_column_) {
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