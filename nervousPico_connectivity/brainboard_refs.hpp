#ifndef BRAINBOARD_REFS_HPP
#define BRAINBOARD_REFS_HPP

#include <stdint.h>

/* NOTE: This is a quick reference of BrainBoard, the things that are commented are things which are not present in the 
FW API. Also not everything is tested yet...  */
namespace BRAINBOARD_HOST {

/* EYES REFS: */
    enum class EyeAnimation : uint8_t { 
        MOVE                = 0,
        BLINK_ANIM          = 1,
        CONFUSED_ANIM       = 2,
        THINKING_ANIM       = 3,
        DISABLE             = 4,
    };

    enum class EyeID : uint8_t {
        LEFT        = 1,
        RIGHT       = 0,
        BOTH        = 2,
    }; 

/* LEDSTRIP REF: */
    enum class LedstripCommandTypes {
        SET_COLOR           = 0,
        ANIM_BLINK          = 1,
        ANIM_FADE_OUT       = 2,
        ANIM_FADE_IN        = 3,
    };

    enum class LedID : uint8_t {
        HEAD        = 1,
        BODY        = 2,
        // BOTH        = 2,
    }; 

    enum class Color {
        RED,
        GREEN,
        BLUE,
        WHITE,
        BLACK
    };

/* Motor REF */
    enum class MotorID : uint8_t {
        AZIMUTH        = 1, // X axis
        ELEVATION      = 2, // Y axis
        // BOTH        = 2,
    }; 

} // namespace SERIAL
#endif // BRAINBOARD_REFS_HPP