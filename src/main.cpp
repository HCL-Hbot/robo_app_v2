#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <atomic>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <openwakeword.hpp>
#include <chrono>
#include <boost/asio.hpp>
#include <person_detector.hpp>
#include <brainboard_host.hpp>
#include <settings.hpp>
#include "common.h"
#include <whisper_helper.hpp>
#include <llama_wrapper.hpp>
#include <whisper_wrapper.hpp>
#include <common-sdl.h>
#include <regex>

static volatile bool is_interrupted = false;

/* Interrupt handler for exiting the program */
void interrupt_handler(int _)
{
    (void)_;
    is_interrupted = true;
}

int main(int argc, char *argv[])
{
    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    /*
     * Whisper_params
     *
     * This is the struct which holds all the settings for the llama and whisper model
     * By default it contains valid settings; But feel free to change any of the settings
     * The definition of this struct is in settings.hpp
     */
    model_params params;

    /* Create instance of whisper_wrapper
     * This class loads in and manages the whisper model
     */
    whisper_wrapper whisper_inst(params);

    /* Create instance of llama_wrapper
     * This class loads in and manages the llama model
     */
    llama_wrapper llama_inst(params);

    /*
     * Create the async audio buffer with a instance of the RTP receiver so that the RTP receiver automatically updates the audio buffer
     */
    audio_async audio(audio_buffer_size_ms);
    if (!audio.init(-1, WHISPER_SAMPLE_RATE))
    {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    // Start the audio capture
    audio.resume();

    /* This function call will load in the whisper model
     * It will also do a test run to calibrate the response of the model
     * The test run also prepares the model for continuous operation
     */
    whisper_inst.init();

    /* This function call will load in the llama model
     * It will also do a test run to calibrate the response of the model (some tokenizening and stuff)
     * The test run also prepares the model for continuous operation
     */
    llama_inst.init();

    /* Buffer which holds the audio sensors that will be captured with the audio.get functions */
    std::vector<float> pcmf32_cur;

    /* Keeps track of new audio samples were coming in previously, it get's compared with current state of audio sampling buffer
     * So that the timer can keep track of it's state and reset when necessary
     */
    bool audio_was_active = false;

    BRAINBOARD_HOST::DeviceController device_controller("/dev/ttyACM0", 1200);
    // PersonDetector persondetect("127.0.0.1", "5678", device_controller);
    // persondetect.init();
    /* End of Person Tracking Subsystem: */

    /* Audio Interaction Subsystem: */
    openwakeword_detector wakeword_detect;
    wakeword_detect.init("../model/hey_robo.onnx");

    // Start timer for Robo blinking:
    auto start_time = std::chrono::steady_clock::now();
    cv::Mat cam_frame;
    // clear audio buffer
    audio.clear();
    while (!is_interrupted)
    {

        /* Non-blocking call to get 32-bit floating point pcm samples from circular audio buffer
         * The default amount of samples to be fetched is around 2000ms, that is enough to fetch one or multiple words
         */
        audio.get(2000, pcmf32_cur);

        bool wake_word_detected = wakeword_detect.detect_wakeword();
        if (wake_word_detected)
        {
            printf("Wake word detected!\n");
            device_controller.controlEyes(BRAINBOARD_HOST::EyeID::BOTH, 0, 0, BRAINBOARD_HOST::EyeAnimation::THINKING_ANIM, 100);
            audio.get(2000, pcmf32_cur);
            // Introduce a 2-second delay after the wake word is detected
            std::this_thread::sleep_for(std::chrono::seconds(2));
  
            audio_was_active = true;
        }
        if (audio_was_active)
        {
            /* Detect audio activity using vad; A algorithm that determines voice activity by applying a band-pass filter and looking at the energy of the audio captured
             * Want to learn more? See: https://speechprocessingbook.aalto.fi/Recognition/Voice_activity_detection.html
             */
            bool voice_activity_detected = ::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1250, params.vad_thold, params.freq_thold, params.print_energy);
            if (voice_activity_detected)
            {

                /* Capture some more audio, to make sure we capture a full sentence, the amount of time to be captured is defined in the model_params struct */
                audio.get(params.voice_ms, pcmf32_cur);

                /* Run inference with whisper on the captured audio, this will output the sentence that was captured from the audio */
                std::string text_heard = whisper_inst.do_inference(pcmf32_cur);
                std::string result = "";
                /* Was there any text returned from the whisper inference? If no text, we obviously don't want to run llama on it!*/
                if (text_heard.empty())
                {
                    /* No words were captured! Continue with capturing new audio; */
                    audio.clear();
                    continue;
                }
                else
                {
                    /* Words were captured! Reset the timeout timer! */
                    audio_was_active = false;
                }
                std::regex pattern(R"(\brobo\b,?)", std::regex_constants::icase); 
                result = std::regex_replace(text_heard, pattern, "");
                /* Print the text we got from whisper inference  */
                fprintf(stdout, "%s%s%s", "\033[1m", result.c_str(), "\033[0m");
                fflush(stdout);

                // /* Run llama inference */
                std::string text_to_speak = llama_inst.do_inference(text_heard);

                // /* Run an external script to convert the text back to speech and stream it back to the client */
                speak_with_file(params.speak, text_to_speak, params.speak_file, 2);

                /* Empty the audio buffer, so that we do not process the same audio twice! */
                audio.clear();
                audio_was_active = false;
            }
        }

        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

        if (elapsed_time > 1000 * 20)
        {
            // Every 30 sec Robo blinks.
            printf("10 second has passed.\n");
            device_controller.blink(BRAINBOARD_HOST::EyeID::BOTH);
            // Reset the start time
            start_time = current_time;
        }
    }
    return 0;
}
