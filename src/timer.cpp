#include "timer.h"

using namespace std::chrono;

void Timer::start() {
    start_time_ = high_resolution_clock::now();
}

void Timer::stop() {
    end_time_ = high_resolution_clock::now();
}

long long Timer::get_seconds() {
    auto duration = duration_cast<seconds>(end_time_ - start_time_);
    return duration.count();
}

long long Timer::get_milliseconds() {
    auto duration = duration_cast<milliseconds>(end_time_ - start_time_);
    return duration.count();
}

long long Timer::get_microseconds() {
    auto duration = duration_cast<microseconds>(end_time_ - start_time_);
    return duration.count();
}

long long Timer::get_nanoseconds() {
    auto duration = duration_cast<nanoseconds>(end_time_ - start_time_);
    return duration.count();
}
