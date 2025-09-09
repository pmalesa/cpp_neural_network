#include <chrono>
#include <string>

using namespace std::chrono;
using std::string;

class Timer {
public:
    Timer();
    Timer(string method_name);
    ~Timer();

private:
    string method_name_;
    high_resolution_clock::time_point start_time_;
};