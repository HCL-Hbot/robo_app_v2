#ifndef PERSON_DETECTOR_HPP
#define PERSON_DETECTOR_HPP
#include <string>
#include <cstdint>
#include <atomic>
#include <thread>
#include <boost/asio.hpp>

#include <brainboard_host.hpp>

typedef struct {
    const int x;
    const int y;
    const int distance;
} Person_t;

class PersonDetector {
public:
    PersonDetector(const std::string& host, const std::string& port, BRAINBOARD_HOST::DeviceController& ref) 
        : host_(host), port_(port), device_controller_(ref) {
            device_controller_.controlEyes(BRAINBOARD_HOST::EyeID::BOTH, 0, 0, BRAINBOARD_HOST::EyeAnimation::MOVE);
        }
    ~PersonDetector();

    void init();
    Person_t record_person_position();
    void update_eye_position();
    const uint8_t map_value(int value, int from_low, int from_high, int to_low, int to_high);

private:
/* Class Constants */
    const static inline int MAX_X = 640; // Display max x in px.
    const static inline int MAX_Y = 480; // Display max y in px.

    const static inline int MAX_EYE_X = 120;
    const static inline int MIN_EYE_X = -120;
    
    const static inline int MAX_EYE_Y = 120;
    const static inline int MIN_EYE_Y = -120;

/* Local TCP Socket Parameters: */
    std::string host_;
    std::string port_;

/* Thread Handler Attributes: */
    std::atomic<int> _distance{0};
    std::atomic<int> x_position{0};
    std::atomic<int> y_position{0};
    std::atomic_bool person_detected{false};
    std::atomic_bool stop_cond{false};
    std::thread _tcp_recv_thread;

    BRAINBOARD_HOST::DeviceController& device_controller_;
};

#endif /* PERSON_DETECTOR_HPP */