#include <person_detector.hpp>
#include <iostream>
#include <chrono>
#include <iostream>
#include <string>
#include <atomic>
#include <thread>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/* LOCAL PROTOTYPES: */
const bool parse_json(const std::string& line, std::atomic<int>& distance_detected, 
    std::atomic<int>& x, std::atomic<int>& y, std::atomic_bool& person_detected);
void person_tracker_thread(const std::string& ip, const std::string& port, 
        std::atomic<int>& distance_detected, std::atomic<int>& x, std::atomic<int>& y, 
        std::atomic_bool& person_detected, std::atomic_bool& stop_cond, PersonDetector* detector);

PersonDetector::~PersonDetector() {
    stop_cond = true;
    _tcp_recv_thread.join();
}

void PersonDetector::init() {
    _tcp_recv_thread = std::thread(person_tracker_thread, host_, port_, 
        std::ref(_distance), std::ref(x_position), std::ref(y_position), 
        std::ref(person_detected), std::ref(stop_cond), this);
}

Person_t PersonDetector::record_person_position() {
    if (person_detected) {
        Person_t new_person_position{x_position, y_position, _distance};
        return new_person_position;
    }
    return {0, 0, 0};
}

void PersonDetector::update_eye_position() {
    int current_target_x = x_position.load();
    int current_target_y = y_position.load();

    int x_movement = 0;
    int y_movement = 0;

    // X position thresholds
    if (current_target_x > 400) {
        x_movement = 20;
    } else if (current_target_x < 250) {
        x_movement = -20;
    }

    // Y position thresholds
    if (current_target_y < 90) {
        y_movement = 20;
    } else if (current_target_y > 220) {
        y_movement = -20;
    }

    device_controller_.move_eye(BRAINBOARD_HOST::EyeID::BOTH, x_movement, y_movement, 200);
}

void person_tracker_thread(const std::string& ip, const std::string& port, 
        std::atomic<int>& distance_detected, std::atomic<int>& x, std::atomic<int>& y, 
        std::atomic_bool& person_detected, std::atomic_bool& stop_cond, PersonDetector* detector) {
    try {
        boost::asio::io_context io_context;
        boost::asio::ip::tcp::resolver resolver(io_context);
        boost::asio::ip::tcp::resolver::results_type endpoints = resolver.resolve(ip, port);

        std::cout << "port: " << port << " ip: " << ip << '\n';

        // Create and connect the socket
        boost::asio::ip::tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints);

        // Buffer to store the response
        boost::asio::streambuf buffer;

        std::vector<int> x_values;
        std::vector<int> y_values;
        std::vector<int> distance_values;

        auto start_time = std::chrono::steady_clock::now();

        while (!stop_cond) {
            boost::asio::read_until(socket, buffer, '\n');

            std::istream is(&buffer);
            std::string line;
            std::getline(is, line);

            buffer.consume(buffer.size());

            if (!line.empty() && line.find_first_not_of(' ') != std::string::npos) {
                if (parse_json(line, distance_detected, x, y, person_detected)) {
                    x_values.push_back(x.load());
                    y_values.push_back(y.load());
                    distance_values.push_back(distance_detected.load());

                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

                    if (elapsed_time >= 15) {
                        int avg_x = std::accumulate(x_values.begin(), x_values.end(), 0) / x_values.size();
                        int avg_y = std::accumulate(y_values.begin(), y_values.end(), 0) / y_values.size();
                        int avg_distance = std::accumulate(distance_values.begin(), distance_values.end(), 0) / distance_values.size();

                        // Update vars
                        x = avg_x;
                        y = avg_y;
                        distance_detected = avg_distance;

                        // clear buffers
                        x_values.clear();
                        y_values.clear();
                        distance_values.clear();

                        detector->update_eye_position();
                        // Reset the start time
                        start_time = std::chrono::steady_clock::now();
                    }
                }
            } else {
                std::cerr << "Received an empty or whitespace-only line, ignoring.\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // socket reading interval
        }
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}


const bool parse_json(const std::string& line, std::atomic<int>& distance_detected, 
    std::atomic<int>& x, std::atomic<int>& y, std::atomic_bool& person_detected) {
    try {
        json parsed = json::parse(line);
        if (parsed.contains("face detected") && parsed.contains("head position")) {
            if (parsed["face detected"].get<int>() == 1) {
                distance_detected = static_cast<int>(parsed["head position"][2].get<double>());
                x = static_cast<int>(parsed["head position"][0].get<double>());
                y = static_cast<int>(parsed["head position"][1].get<double>());
                person_detected = true;
                std::cout << "X: " << x << ", Y: " << y << std::endl;
                return true;
            } else {
                person_detected = false;
            }
        }
    } catch (json::parse_error& e) {
        std::cerr << "Parse error: " << e.what() << " - Message: " << line << std::endl;
    }
    return false;
}

const uint8_t PersonDetector::map_value(int value, int from_low, int from_high, int to_low, int to_high) {
    return static_cast<uint8_t>((value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low);
}