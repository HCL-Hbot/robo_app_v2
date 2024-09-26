#ifndef SERIAL_DRIVER_HPP
#define SERIAL_DRIVER_HPP

#include <string>
#include <iostream>
#include <libserialport.h>
#include <serial_settings.hpp>

namespace SERIAL {
class SerialDriver {
public:
    SerialDriver(const std::string& port_name, int baud_rate)
        : port_name_(port_name), baud_rate_(baud_rate), port_(nullptr) {}

    ~SerialDriver() {
        close();
    }

    void open() {
        check(sp_get_port_by_name(port_name_.c_str(), &port_));
        check(sp_open(port_, SP_MODE_READ_WRITE));
        check(sp_set_baudrate(port_, baud_rate_));
    }

    void close() {
        if (port_) {
            check(sp_close(port_));
            sp_free_port(port_);
            port_ = nullptr;
        }
    }

    void write(const std::string& data) {
        int size = data.size();
        int bytes_written = sp_blocking_write(port_, data.c_str(), size, 1000);
        if (bytes_written < 0) {
            check(bytes_written);  // Handle the error if any
        } else if (bytes_written < size) {
            std::cerr << "Error: Only " << bytes_written << " bytes written out of " << size << std::endl;
        } else {
            std::cout << "Data sent successfully." << std::endl;
        }
        check(sp_drain(port_));  // Ensure all written bytes are transmitted
    }

    std::string read() {
        char buffer[1024];
        int bytes_read = sp_blocking_read(port_, buffer, sizeof(buffer) - 1, 1000);
        if (bytes_read < 0) {
            check(bytes_read);  // Handle the error if any
        }
        buffer[bytes_read] = '\0';
        return std::string(buffer);
    }

private:
    std::string port_name_;
    int baud_rate_;
    struct sp_port* port_;

    void check(int result) {
        if (result != SP_OK) {
            const char* error_message = sp_last_error_message();
            std::cerr << "Error: " << error_message << std::endl;
            sp_free_error_message((char*)error_message);
            exit(EXIT_FAILURE);
        }
    }
};
} // NAMESPACE SERIAL 
#endif // End of SERIAL_DRIVER_HPP