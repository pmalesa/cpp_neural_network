#include "dataset.h"
#include <gtest/gtest.h>
#include "matrix.h"
#include "logger.h"
#include <stdexcept>

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
};

TEST_F(DatasetTest, EmptyMethodTest) {
    Dataset dataset;
    EXPECT_TRUE(dataset.empty());
}

TEST_F(DatasetTest, GetRangeMethodTest) {

}

TEST_F(DatasetTest, AccessOperatorTest) {

}

TEST_F(DatasetTest, SizeTest) {

}

