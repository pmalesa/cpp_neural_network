#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include "matrix.h"

using std::string;
using std::vector;

class Dataset {
public:
    Dataset();
    Dataset(const string& filename);
    ~Dataset();

    void load_csv(const string& filename, bool headers = false, bool index_column = false);
    void save_csv(const string& filename);

    Matrix operator[](size_t row) const;
    Matrix get_data() const { return data_; }
    Matrix get_range(size_t start, size_t end) const;
    const string& get_row(size_t row) const;
    vector<string> get_headers() const { return header_names_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    Matrix data_;
    vector<string> rows_;
    vector<string> header_names_;
    size_t size_;
    bool headers_;
    bool index_column_;
    bool categorical_;
};

#endif // DATASET_H