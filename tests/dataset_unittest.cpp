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
    bool string_row_equal_(const vector<string>& lhs, const vector<string>& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }

        return true;
    }

    bool double_row_equal_(const vector<double>& lhs, const vector<double>& rhs, double tolerance = 1e-9) {
        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); ++i) {
            if (abs(lhs[i] - rhs[i]) > tolerance) {
                return false;
            }
        }

        return true;
    }
};

TEST_F(DatasetTest, EmptyMethodTest) {
    Dataset ds;
    EXPECT_TRUE(ds.empty());
}

TEST_F(DatasetTest, LoadCSVWithHeadersTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test.csv", true);

    // Test header
    vector<string> expected_headers = { "sepal.length", "sepal.width", "petal.length", "petal.width", "variety" };
    EXPECT_TRUE(string_row_equal_(ds.get_headers(), expected_headers));

    // Test raw data
    const auto& raw_data = ds.get_raw_data();
    ASSERT_TRUE(raw_data.size() == 150);
    EXPECT_TRUE(string_row_equal_(raw_data[0], { "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(string_row_equal_(raw_data[75], { "6.6", "3", "4.4", "1.4", "Versicolor" }));
    EXPECT_TRUE(string_row_equal_(raw_data[149], { "5.9", "3", "5.1", "1.8", "Virginica" }));
    
    // Test numerical data
    Matrix data = ds.get_data();
    ASSERT_TRUE(data.get_rows() == 150);
    EXPECT_TRUE(double_row_equal_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(double_row_equal_(data[75], { 6.6, 3, 4.4, 1.4 }));
    EXPECT_TRUE(double_row_equal_(data[149], { 5.9, 3, 5.1, 1.8 }));

    // Test numerical target values
    Matrix targets = ds.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 0);
    EXPECT_TRUE(static_cast<int>(targets[75][0]) == 1);
    EXPECT_TRUE(static_cast<int>(targets[149][0]) == 2);
}

TEST_F(DatasetTest, LoadCSVCorruptedFileTest) {
    Dataset ds;
    EXPECT_THROW(ds.load_csv("./tests/data/dataset_test_corrupted.csv", true), std::invalid_argument);
}

TEST_F(DatasetTest, LoadCSVNumericalTargetsTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test_numerical_targets.csv", true);
    Matrix targets = ds.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 444);
    EXPECT_TRUE(static_cast<int>(targets[75][0]) == 555);
    EXPECT_TRUE(static_cast<int>(targets[149][0]) == 666);
}

TEST_F(DatasetTest, LoadCSVWithIndexColumnTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test_index_column.csv", true, true);
    
    const auto& raw_data = ds.get_raw_data();
    const Matrix& data = ds.get_data();

    // Test raw data
    ASSERT_TRUE(raw_data.size() == 12);
    EXPECT_TRUE(string_row_equal_(raw_data[0], { "1", "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(string_row_equal_(raw_data[4], { "5", "7", "3.2", "4.7", "1.4", "Versicolor" }));
    EXPECT_TRUE(string_row_equal_(raw_data[9], { "10", "7.2", "3", "5.8", "1.6", "Virginica" }));
    
    // Test numerical data
    ASSERT_TRUE(data.get_rows() == 12);
    EXPECT_TRUE(double_row_equal_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(double_row_equal_(data[4], { 7.0, 3.2, 4.7, 1.4 }));
    EXPECT_TRUE(double_row_equal_(data[9], { 7.2, 3.0, 5.8, 1.6 }));

    // Test numerical target values
    const Matrix& targets = ds.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 0);
    EXPECT_TRUE(static_cast<int>(targets[4][0]) == 1);
    EXPECT_TRUE(static_cast<int>(targets[9][0]) == 2);
}

TEST_F(DatasetTest, LoadCSVWithoutHeadersTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test_no_headers.csv");
    
    // Test header names
    EXPECT_TRUE(ds.get_headers().empty());

    const auto& raw_data = ds.get_raw_data();
    const Matrix& data = ds.get_data();

    // Test raw data
    ASSERT_TRUE(raw_data.size() == 150);
    EXPECT_TRUE(string_row_equal_(raw_data[0], { "5.1", "3.5", "1.4", ".2", "Setosa" }));
    EXPECT_TRUE(string_row_equal_(raw_data[75], { "6.6", "3", "4.4", "1.4", "Versicolor" }));
    EXPECT_TRUE(string_row_equal_(raw_data[149], { "5.9", "3", "5.1", "1.8", "Virginica" }));

    // Test numerical data
    ASSERT_TRUE(data.get_rows() == 150);
    EXPECT_TRUE(double_row_equal_(data[0], { 5.1, 3.5, 1.4, 0.2 }));
    EXPECT_TRUE(double_row_equal_(data[75], { 6.6, 3, 4.4, 1.4 }));
    EXPECT_TRUE(double_row_equal_(data[149], { 5.9, 3, 5.1, 1.8 }));

    // Test numerical target values
    Matrix targets = ds.get_targets();
    EXPECT_TRUE(static_cast<int>(targets[0][0]) == 0);
    EXPECT_TRUE(static_cast<int>(targets[75][0]) == 1);
    EXPECT_TRUE(static_cast<int>(targets[149][0]) == 2);
}

TEST_F(DatasetTest, SaveCSVTest) {
    Dataset ds_1;
    ds_1.load_csv("./tests/data/dataset_test.csv", true);

    const string temp_path = "./tests/data/dataset_test_temp.csv";
    ds_1.save_csv(temp_path);
    
    Dataset ds_2;
    ds_2.load_csv(temp_path, true);
    std::remove(temp_path.c_str());

    ASSERT_TRUE(ds_1.size() == ds_2.size());
    ASSERT_TRUE(ds_1.get_columns_count() == ds_2.get_columns_count());
    ASSERT_TRUE(ds_1.get_features_count() == ds_2.get_features_count());

    EXPECT_TRUE(string_row_equal_(ds_1.get_headers(), ds_2.get_headers()));
    EXPECT_TRUE(string_row_equal_(ds_1.get_raw_targets(), ds_2.get_raw_targets()));
    EXPECT_TRUE(ds_1.get_raw_data().size() == ds_2.get_raw_data().size());
    for (size_t row = 0; row < ds_1.get_raw_data().size(); ++row) {
        EXPECT_TRUE(string_row_equal_(ds_1.get_raw_data()[row], ds_2.get_raw_data()[row]));
    }
    EXPECT_TRUE(ds_1.get_data() == ds_2.get_data());
    EXPECT_TRUE(ds_1.get_targets() == ds_2.get_targets());
}

TEST_F(DatasetTest, GetRangeMethodTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test.csv", true);

    // const & are not used here to ensure that different underlying memory is used
    Matrix data = ds.get_data();
    Matrix range = ds.get_range(3, 6);

    ASSERT_TRUE(range.get_rows() == 3);
    ASSERT_TRUE(range.get_cols() == data.get_cols());

    EXPECT_TRUE(double_row_equal_(range[0], data[3]));
    EXPECT_TRUE(double_row_equal_(range[1], data[4]));
    EXPECT_TRUE(double_row_equal_(range[2], data[5]));

    EXPECT_THROW(ds.get_range(3, 2), std::invalid_argument);
}

TEST_F(DatasetTest, AccessOperatorTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test.csv", true);

    EXPECT_TRUE((ds[0] == il{ { 5.1, 3.5, 1.4, 0.2 }}));
    EXPECT_TRUE((ds[75] == il{ { 6.6, 3, 4.4, 1.4 }}));
    EXPECT_TRUE((ds[149] == il{ { 5.9, 3, 5.1, 1.8 } }));
}

TEST_F(DatasetTest, SizeTest) {
    Dataset ds;
    ds.load_csv("./tests/data/dataset_test.csv", true);
    EXPECT_TRUE(ds.size() == 150);
    
    ds.load_csv("./tests/data/dataset_test_index_column.csv", true, true);
    EXPECT_TRUE(ds.size() == 12);
}
