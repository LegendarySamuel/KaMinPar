/*******************************************************************************
 * @file:   strings.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper functions for string operations.
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cctype>
#include <concepts>
#include <string>

namespace kaminpar::utility {
namespace str {
std::string extract_basename(const std::string &path);
std::string to_lower(std::string arg);
std::vector<std::string> explode(const std::string &str, char del);

std::string &rtrim(std::string &s, const char *t = " \t\n\r\f\v");
std::string &ltrim(std::string &s, const char *t = " \t\n\r\f\v");
std::string &trim(std::string &s, const char *t = " \t\n\r\f\v");
} // namespace str
} // namespace kaminpar::utility