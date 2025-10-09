#include "logger.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <string>

using std::cout;
using std::string;

Logger::Logger(Level print_level) : print_level_(print_level), printing_enabled_(true), writing_enabled_(true) {
    generate_log_filename_();
    create_log_file_();
}

void Logger::log(string message, Level level) const {
    if (!printing_enabled_ && !writing_enabled_) {
        return;
    }

    string log_line = construct_log_line_(level, message);

    if (writing_enabled_) {
        append_log_file_(log_line);
    }

    if (printing_enabled_) {
        if (print_level_ >= level) {
            cout << log_line << '\n';
        }
    }
}

bool Logger::is_enabled() const { 
    return printing_enabled_ || writing_enabled_; 
}

void Logger::set_enabled(bool value) { 
    printing_enabled_ = value; 
    writing_enabled_ = value;
}

void Logger::generate_log_filename_() {
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // Format timestamp
    std::tm tm = *std::localtime(&now_time);
    std::ostringstream oss;
    oss << "log_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".txt";
    log_filename_ = oss.str();
}

void Logger::create_log_file_() const {
    std::ofstream log_file(log_filename_, std::ios::out | std::ios::trunc);
    if (!log_file) {
        throw std::runtime_error("Failed to create log file: " + log_filename_);
    }
    log_file << "===== LOG STARTED =====\n";
}

string Logger::construct_log_line_(Level level, string message) const {
    std::ostringstream log_line_oss;

    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time);

    // Construct the single log line
    log_line_oss << "[" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << "]";
    log_line_oss << "[" << level_to_str_(level) << "]: " ;
    log_line_oss << message;

    return log_line_oss.str();
}

void Logger::append_log_file_(string log_line) const {
    std::ofstream log_file(log_filename_, std::ios::app);
    if (!log_file) {
        throw std::runtime_error("Failed to open log file: " + log_filename_);
    }
    log_file << log_line << std::endl;
}

string Logger::level_to_str_(Level level) const {
    auto it = level_to_str_map_.find(level);
    return (it != level_to_str_map_.end()) ? it->second : "UNKNOWN";
}