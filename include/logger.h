#ifndef LOGGER_H
#define LOGGER_H

#include <unordered_map>
#include <string>

using std::string;
using std::unordered_map;

class Logger {
public:
    enum class Level { Critical = 4, Error = 3, Warning = 2, Debug = 1, Info = 0 };

    // Access global singleton instance
    static Logger& instance() {
        static Logger instance_;
        return instance_;
    }
    
    // Logging interface
    void log(string message, Level level = Level::Info) const;
    
    // Logging level management
    Level get_level() const { return print_level_; }
    void set_level(Level new_level) { print_level_ = new_level; };
   
    // Enable/disable options
    bool is_enabled() const;
    void set_enabled(bool value);
    bool is_printing_enabled() const { return printing_enabled_; }
    void set_printing_enabled(bool value) { printing_enabled_ = value; }
    bool is_writing_enabled() const { return writing_enabled_; }
    void set_writing_enabled(bool value) { writing_enabled_ = value; }
    
    ~Logger() = default;

private:
    Logger(Level print_level = Level::Info); // Prevent manual creation of new instances

    // Prevent copying and assignment
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void generate_log_filename_();
    void create_log_file_() const;
    void append_log_file_(string log_line) const;
    string construct_log_line_(Level level, string message) const;
    string level_to_str_(Level level) const;

    Level print_level_;
    string log_filename_;
    bool printing_enabled_;
    bool writing_enabled_;

    static inline const unordered_map<Level, string> level_to_str_map_ = {
        {Level::Critical, "CRITICAL" },
        {Level::Error, "ERROR" },
        {Level::Warning, "WARNING" },
        {Level::Debug, "DEBUG" },
        {Level::Info, "INFO" }
    };
};

#endif // LOGGER_H
