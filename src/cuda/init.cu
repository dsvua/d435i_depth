#include "init.h"
#include <iostream>             // for cout

void error2(const char *error_string, const char *file, const int line,
    const char *func) {
std::cout << "Error: " << error_string << "\t" << file << ":" << line
     << std::endl;
exit(0);
}
