#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "matrix.h"

using std::string;

class Dataset {
public:
    Dataset();
    Dataset(const string& filename);
    ~Dataset();

    void load_csv(const string& filename);
    void save_csv(const string& filename);

private:
    Matrix data_;

};

#endif // DATASET_H