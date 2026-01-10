#include "dataset.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "logger.h"
#include <stdexcept>
#include <cstdio>
#include <iostream>

using std::string;
using il = std::initializer_list<std::initializer_list<double>>;

static Logger& logger = Logger::instance();

class DatasetTest : public testing::Test {
public:
    DatasetTest() {
        logger.set_printing_enabled(false);
    }

    ~DatasetTest() {
        logger.set_printing_enabled(true);
    }

protected:
    bool test_string_row_data_(const vector<string>& line_to_test, const vector<string>& correct_line) {
        if (line_to_test.size() != correct_line.size()) {
            return false;
        }

        for (size_t i = 0; i < line_to_test.size(); ++i) {
            if (line_to_test[i] != correct_line[i]) {
                return false;
            }
        }

        return true;
    }

    bool test_numerical_row_data_(const vector<double>& data_to_test, const vector<double>& true_data) {
        if (data_to_test.size() != true_data.size()) {
            return false;
        }

        for (size_t i = 0; i < data_to_test.size(); ++i) {
            if (abs(data_to_test[i] - true_data[i]) > 1e9) {
                return false;
            }
        }

        return true;
    }
};

TEST_F(DatasetTest, EmptyMethodTest) {
    Dataset dataset;
    EXPECT_TRUE(dataset.empty());
}

TEST_F(DatasetTest, LoadCSVTestOne) {
    // Test on CSV data with header row and without index column
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test.csv", true);
    vector<vector<string>> raw_data = dataset.get_raw_data();
    Matrix data = dataset.get_data();
    
    // Test header names
    vector<string> headers = dataset.get_headers();
    vector<string> headers_true = { "sepal.length", "sepal.width", "petal.length", "petal.width", "variety" };
    EXPECT_TRUE(headers.size() == headers_true.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        EXPECT_TRUE(headers[i] == headers_true[i]);
    }

    // Test raw data
    EXPECT_TRUE(raw_data.size() == 150);
    EXPECT_TRUE(test_string_row_data_(raw_data[0], { "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(test_string_row_data_(raw_data[75], { "6.6", "3", "4.4", "1.4", "Versicolor" }));
    EXPECT_TRUE(test_string_row_data_(raw_data[149], { "5.9", "3", "5.1", "1.8", "Virginica" }));

    // Test numerical data
    EXPECT_TRUE(data.get_rows() == 150);
    EXPECT_TRUE(test_numerical_row_data_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(test_numerical_row_data_(data[75], { 6.6, 3, 4.4, 1.4 }));
    EXPECT_TRUE(test_numerical_row_data_(data[149], { 5.9, 3, 5.1, 1.8 }));

    // Test numerical target values
    Matrix targets = dataset.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 0);
    EXPECT_TRUE(static_cast<int>(targets[75][0]) == 1);
    EXPECT_TRUE(static_cast<int>(targets[149][0]) == 2);
}

TEST_F(DatasetTest, LoadCSVTestTwo) {
    // Test on a corrupted CSV file.
    Dataset dataset;
    EXPECT_THROW(dataset.load_csv("./tests/data/dataset_test_corrupted.csv", true), std::invalid_argument);
}

TEST_F(DatasetTest, LoadCSVTestThree) {
    // Test on data with numerical target values
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test_numerical_targets.csv", true);
    Matrix targets = dataset.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 444);
    EXPECT_TRUE(static_cast<int>(targets[75][0]) == 555);
    EXPECT_TRUE(static_cast<int>(targets[149][0]) == 666);
}

TEST_F(DatasetTest, LoadCSVTestFour) {
    // Test on data with an index column
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test_index_column.csv", true, true);
    vector<vector<string>> raw_data = dataset.get_raw_data();
    Matrix data = dataset.get_data();

    // Test raw data
    EXPECT_TRUE(raw_data.size() == 12);
    EXPECT_TRUE(test_string_row_data_(raw_data[0], { "1", "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(test_string_row_data_(raw_data[4], { "5", "7", "3.2", "4.7", "1.4", "Versicolor" }));
    EXPECT_TRUE(test_string_row_data_(raw_data[9], { "10", "7.2", "3", "5.8", "1.6", "Virginica" }));
    
    // Test numerical data
    EXPECT_TRUE(data.get_rows() == 12);
    EXPECT_TRUE(test_numerical_row_data_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(test_numerical_row_data_(data[4], { 7.0, 3.2, 4.7, 1.4 }));
    EXPECT_TRUE(test_numerical_row_data_(data[9], { 7.2, 3.0, 5.8, 1.6 }));

    // Test numerical target values
    Matrix targets = dataset.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 0);
    EXPECT_TRUE(static_cast<int>(targets[4][0]) == 1);
    EXPECT_TRUE(static_cast<int>(targets[9][0]) == 2);
}

TEST_F(DatasetTest, LoadCSVTestFive) {
    // Test on data without headers
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test_no_headers.csv");
    vector<vector<string>> raw_data = dataset.get_raw_data();
    Matrix data = dataset.get_data();
    
    // Test header names
    vector<string> headers = dataset.get_headers();
    EXPECT_TRUE(headers.empty());

    // Test raw data
    EXPECT_TRUE(raw_data.size() == 150);
    EXPECT_TRUE(test_string_row_data_(raw_data[0], { "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(test_string_row_data_(raw_data[75], { "6.6", "3", "4.4", "1.4", "Versicolor" }));
    EXPECT_TRUE(test_string_row_data_(raw_data[149], { "5.9", "3", "5.1", "1.8", "Virginica" }));

    // Test numerical data
    EXPECT_TRUE(data.get_rows() == 150);
    EXPECT_TRUE(test_numerical_row_data_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(test_numerical_row_data_(data[75], { 6.6, 3, 4.4, 1.4 }));
    EXPECT_TRUE(test_numerical_row_data_(data[149], { 5.9, 3, 5.1, 1.8 }));

    // Test numerical target values
    Matrix targets = dataset.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 0);
    EXPECT_TRUE(static_cast<int>(targets[75][0]) == 1);
    EXPECT_TRUE(static_cast<int>(targets[149][0]) == 2);
}

TEST_F(DatasetTest, SaveCSVTestOne) {
    Dataset dataset_1, dataset_2;
    dataset_1.load_csv("./tests/data/dataset_test.csv", true);
    string temp_filepath = "./tests/data/dataset_test_temp.csv";
    dataset_1.save_csv(temp_filepath);
    dataset_2.load_csv(temp_filepath, true);
    std::remove(temp_filepath.c_str());

    EXPECT_TRUE(dataset_1.size() == dataset_2.size());
    EXPECT_TRUE(dataset_1.get_columns_count() == dataset_2.get_columns_count());
    EXPECT_TRUE(dataset_1.get_features_count() == dataset_2.get_features_count());

    vector<vector<string>> raw_data_1 = dataset_1.get_raw_data();
    vector<string> raw_targets_1 = dataset_1.get_raw_targets();
    vector<string> headers_1 = dataset_1.get_headers();
    Matrix data_1 = dataset_1.get_data();
    Matrix targets_1 = dataset_1.get_targets();

    vector<vector<string>> raw_data_2 = dataset_2.get_raw_data();
    vector<string> raw_targets_2 = dataset_2.get_raw_targets();
    vector<string> headers_2 = dataset_2.get_headers();
    Matrix data_2 = dataset_2.get_data();
    Matrix targets_2 = dataset_2.get_targets();

    EXPECT_TRUE(raw_data_1.size() == raw_data_2.size());
    for (size_t row = 0; row < raw_data_1.size(); ++row) {
        EXPECT_TRUE(test_string_row_data_(raw_data_1[row], raw_data_2[row]));
    }
    EXPECT_TRUE(test_string_row_data_(raw_targets_1, raw_targets_2));
    EXPECT_TRUE(test_string_row_data_(headers_1, headers_2));
    EXPECT_TRUE(data_1 == data_2);
    EXPECT_TRUE(targets_1 == targets_2);
}

TEST_F(DatasetTest, GetRangeMethodTest) {
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test.csv", true);
    vector<vector<string>> raw_data = dataset.get_raw_data();
    Matrix data = dataset.get_data();
    Matrix data_range = dataset.get_range(3, 6);

    EXPECT_TRUE(data_range.get_rows() == 3);
    EXPECT_TRUE(data_range.get_cols() == data.get_cols());
    EXPECT_TRUE(test_numerical_row_data_(data_range[0], data[3]));
    EXPECT_TRUE(test_numerical_row_data_(data_range[1], data[4]));
    EXPECT_TRUE(test_numerical_row_data_(data_range[2], data[5]));
    EXPECT_THROW(dataset.get_range(3, 2), std::invalid_argument);
}

TEST_F(DatasetTest, AccessOperatorTest) {
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test.csv", true);
    EXPECT_TRUE((dataset[0] == il{ { 5.1, 3.5, 1.4, 0.2 }}));
    EXPECT_TRUE((dataset[75] == il{ { 6.6, 3, 4.4, 1.4 }}));
    EXPECT_TRUE((dataset[149] == il{ { 5.9, 3, 5.1, 1.8 } }));
}

TEST_F(DatasetTest, SizeTest) {
    Dataset dataset;
    dataset.load_csv("./tests/data/dataset_test.csv", true);
    EXPECT_TRUE(dataset.size() == 150);
    dataset.load_csv("./tests/data/dataset_test_index_column.csv", true, true);
}

