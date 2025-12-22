#include "dataset.h"

Dataset::Dataset()
    : size_(0), headers_(false), index_column_(false), categorical_(false) {

}

Dataset::Dataset(const string& filename)
    : size_(0), headers_(false), index_column_(false), categorical_(false) {

}

Dataset::~Dataset() {

}

void Dataset::load_csv(const string& filename, bool headers, bool index_column) {

}

void Dataset::save_csv(const string& filename) {
    
}

Matrix Dataset::operator[](size_t row) const {
    return data_[row];
}

Matrix Dataset::get_range(size_t start, size_t end) const {
    // TODO
    // Return a range from start to end as Matrix object (inlcusive)

    return data_[end];
}


const string& Dataset::get_row(size_t row) const {
    return rows_[row];
}
