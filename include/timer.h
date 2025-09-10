#include <chrono>

using namespace std::chrono;

class Timer {
public:
    Timer() = default;
    ~Timer() = default;

    void start();
    void stop();

    long long get_seconds();
    long long get_milliseconds();
    long long get_microseconds();
    long long get_nanoseconds();

private:
    high_resolution_clock::time_point start_time_;
    high_resolution_clock::time_point end_time_;
};