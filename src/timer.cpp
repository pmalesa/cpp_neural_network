#include "timer.h"
#include <iostream>

using std::cout;

Timer::Timer() 
    : method_name_(""), start_time_(high_resolution_clock::now()) { }

Timer::Timer(string method_name) 
    : method_name_(method_name), start_time_(high_resolution_clock::now()) { }

Timer::~Timer() {
    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time_);
    if (method_name_.empty()) {
        cout << ">>> Execution time: " << duration.count() << "ms\n";
    } else {
        cout << ">>> Method '"<< method_name_ << "' >>> Execution time: " << duration.count() << "ms\n";
    }
}