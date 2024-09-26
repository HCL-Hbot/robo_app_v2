#ifndef SERIAL_SETTINGS_HPP
#define SERIAL_SETTINGS_HPP

#include <stdint.h>

namespace SERIAL {
    static constexpr uint32_t BAUD_RATE = 115200;   
    static const char* PORT_NAME = "/dev/serial0";
} // namespace SERIAL
#endif // SERIAL_SETTINGS_HPP