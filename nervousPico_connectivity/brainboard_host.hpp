#ifndef DEVICE_CONTROLLER_HPP
#define DEVICE_CONTROLLER_HPP

#include <sstream>
#include <iomanip>
#include <serial_driver.hpp>
#include <brainboard_refs.hpp>

namespace BRAINBOARD_HOST {

class DeviceController {
public:
    DeviceController(const std::string& port_name = SERIAL::PORT_NAME, int baud_rate = SERIAL::BAUD_RATE)
        : serial_port_(port_name, baud_rate) {
        serial_port_.open();
    }

    ~DeviceController() {
        serial_port_.close();
    }

    void controlEyes(const EyeID eye_nr, int x, int y, EyeAnimation anim, int duration = 1000) {
        std::ostringstream command;
        command << "DS101 " << static_cast<int>(eye_nr) << " " << (x) << " " << (y) << " " << duration << "\r\n";
        serial_port_.write(command.str());
        handleEyeAnimation(eye_nr, anim);
    }

    void move_eye(const EyeID eye_nr, int x, int y, int duration = 100) {
        std::ostringstream command;
        command << "DS101 " << static_cast<int>(eye_nr) << " " << (x) << " " << (y) << " " << duration << "\r\n";
        serial_port_.write(command.str());
    }

    void blink(const EyeID eye_nr) {
        std::ostringstream command;
        handleEyeAnimation(eye_nr, EyeAnimation::BLINK_ANIM);
    }

    void disable_think_anim() {
        serial_port_.write("DS106\r\n");
    }

    void controlLedstrip(LedID strip_nr, int hue, int saturation, int value, LedstripCommandTypes anim, int duration = 1000) {
        int r, g, b;
        hsv_to_rgb(hue, saturation, value, r, g, b);

        std::ostringstream command;
        command << "LS101 " << static_cast<int>(strip_nr) << " " << r << " " << g << " " << b << "\r\n";
        serial_port_.write(command.str());

        if (anim == LedstripCommandTypes::ANIM_BLINK) {
            command.str(""); // Clear the stream
            command << "LS102 " << static_cast<int>(strip_nr) << " " << r << " " << g << " " << b << " " << duration << "\r\n";
            serial_port_.write(command.str());
        } else if (anim == LedstripCommandTypes::ANIM_FADE_IN) {
            command.str("");
            command << "LS104 " << static_cast<int>(strip_nr) << " " << duration << "\r\n";
            serial_port_.write(command.str());
        } else if (anim == LedstripCommandTypes::ANIM_FADE_OUT) {
            command.str("");
            command << "LS103 " << static_cast<int>(strip_nr) << " " << duration << "\r\n";
            serial_port_.write(command.str());
        } else if (anim == LedstripCommandTypes::SET_COLOR) {
            // No additional command needed for solid color
        } else {
            std::cerr << "Animation anim is not implemented" << std::endl;
        }
    }

    void setColor(LedID strip_nr, Color color, LedstripCommandTypes anim = LedstripCommandTypes::SET_COLOR, int duration = 1000) {
        int r, g, b;
        switch (color) {
            case Color::RED:
                r = 255; g = 0; b = 0;
                break;
            case Color::GREEN:
                r = 0; g = 255; b = 0;
                break;
            case Color::BLUE:
                r = 0; g = 0; b = 255;
                break;
            case Color::WHITE:
                r = 255; g = 255; b = 255;
                break;
            case Color::BLACK:
                r = 0; g = 0; b = 0;
                break;
            default:
                r = 255; g = 255; b = 255;
                break;
        }

        std::ostringstream command;
        command << "LS101 " << static_cast<int>(strip_nr) << " " << r << " " << g << " " << b << "\r\n";
        serial_port_.write(command.str());

        if (anim == LedstripCommandTypes::ANIM_BLINK) {
            command.str(""); // Clear the stream
            command << "LS102 " << static_cast<int>(strip_nr) << " " << r << " " << g << " " << b << " " << duration << "\r\n";
            serial_port_.write(command.str());
        } else if (anim == LedstripCommandTypes::ANIM_FADE_IN) {
            command.str("");
            command << "LS104 " << static_cast<int>(strip_nr) << " " << duration << "\r\n";
            serial_port_.write(command.str());
        } else if (anim == LedstripCommandTypes::ANIM_FADE_OUT) {
            command.str("");
            command << "LS103 " << static_cast<int>(strip_nr) << " " << duration << "\r\n";
            serial_port_.write(command.str());
        } else if (anim == LedstripCommandTypes::SET_COLOR) {
            // No additional command needed for solid color
        } else {
            std::cerr << "Animation anim is not implemented" << std::endl;
        }
    }

    void controlHeadMovement(MotorID id, int speed, int angle) {
        std::ostringstream command;
        command << "M101 " << static_cast<int>(id) << " " << speed << "\r\n";
        serial_port_.write(command.str());
        command.str("");
        command << "M102 " << static_cast<int>(id) << " " << angle << "\r\n";
        serial_port_.write(command.str());
        command.str("");
    }

private:
    SERIAL::SerialDriver serial_port_;

    void handleEyeAnimation(EyeID eye_nr, EyeAnimation anim) {
        std::ostringstream command;
        switch (anim) {
            case EyeAnimation::BLINK_ANIM:
                command << "DS102 " << static_cast<int>(eye_nr) << " 500\r\n";
                break;
            case EyeAnimation::CONFUSED_ANIM:
                command << "DS103 " << static_cast<int>(eye_nr) << " 500\r\n";
                break;
            case EyeAnimation::THINKING_ANIM:
                command << "DS104 " << static_cast<int>(eye_nr) << " 500\r\n";
                break;
            case EyeAnimation::MOVE:
                std::cerr << "No Animation just move\r\n";
                return; 
            default:
                std::cerr << "Animation not implemented\r\n";
                return;
        }
        serial_port_.write(command.str());
    }

    void hsv_to_rgb(int h, int s, int v, int& r, int& g, int& b) {
        double dh = h / 360.0;
        double ds = s / 100.0;
        double dv = v / 100.0;
        int i = static_cast<int>(dh * 6);
        double f = dh * 6 - i;
        double p = dv * (1 - ds);
        double q = dv * (1 - f * ds);
        double t = dv * (1 - (1 - f) * ds);

        switch (i % 6) {
            case 0: r = dv * 255, g = t * 255, b = p * 255; break;
            case 1: r = q * 255, g = dv * 255, b = p * 255; break;
            case 2: r = p * 255, g = dv * 255, b = t * 255; break;
            case 3: r = p * 255, g = q * 255, b = dv * 255; break;
            case 4: r = t * 255, g = p * 255, b = dv * 255; break;
            case 5: r = dv * 255, g = p * 255, b = q * 255; break;
        }
    }

}; // Class Brainboard_host
} // End of NAMESPACE BRAINBOARD_HOST
#endif // DEVICE_CONTROLLER_HPP