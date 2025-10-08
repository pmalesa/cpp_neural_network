#ifndef LOGGER_H
#define LOGGER_H

#include <unordered_map>
#include <string>

using std::string;
using std::unordered_map;

class Logger {
public:
    enum class Level { Critical = 4, Error = 3, Warning = 2, Debug = 1, Info = 0 };

    Logger(Level print_level = Level::Info);
    ~Logger() = default;

    void log(string message, Level level = Level::Info);
    Level get_level() { return print_level_; }
    void set_level(Level new_level) { print_level_ = new_level; };

private:
    void generate_log_filename_();
    void create_log_file_();
    void append_log_file_(string log_line);
    string construct_log_line_(Level level, string message);
    string level_to_str_(Level level);

    Level print_level_;
    string log_filename_;

    static inline const unordered_map<Level, string> level_to_str_map_ = {
        {Level::Critical, "CRITICAL" },
        {Level::Error, "ERROR" },
        {Level::Warning, "WARNING" },
        {Level::Debug, "DEBUG" },
        {Level::Info, "INFO" }
    };
};

#endif // LOGGER_H
