#include "dataset.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "logger.h"
#include <stdexcept>
#include <iostream>

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
    bool test_splitted_row_data_(const vector<string>& line_to_test, const vector<string>& correct_line) {
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
    EXPECT_TRUE(test_splitted_row_data_(raw_data[0], { "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(test_splitted_row_data_(raw_data[75], { "6.6", "3", "4.4", "1.4", "Versicolor" }));
    EXPECT_TRUE(test_splitted_row_data_(raw_data[149], { "5.9", "3", "5.1", "1.8", "Virginica" }));

    // Test numerical data
    
    EXPECT_TRUE(data.get_rows() == 150);
    EXPECT_TRUE(test_numerical_row_data_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(test_numerical_row_data_(data[75], { 6.6, 3, 4.4, 1.4 }));
    EXPECT_TRUE(test_numerical_row_data_(data[149], { 5.9, 3, 5.1, 1.8 }));

    // Test numerical target values
    // ...
}

TEST_F(DatasetTest, LoadCSVTestTwo) {
    // Test on a corrupted CSV file.
    Dataset dataset;
    EXPECT_THROW(dataset.load_csv("./tests/data/dataset_test_corrupted.csv", true), std::invalid_argument);
}

TEST_F(DatasetTest, GetRangeMethodTest) {

}

TEST_F(DatasetTest, AccessOperatorTest) {

}

TEST_F(DatasetTest, SizeTest) {

}

