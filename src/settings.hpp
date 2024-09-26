#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <cstdint>
#include <string.h>
#include <thread>
#include <algorithm>


/* Time to wait for audio responses coming in*/
const float timeout_time = 10.0f;

/* 
* Size of the circular audio buffer in milliseconds
* Default is 30 seconds, change if needed 
*/
static inline constexpr uint32_t audio_buffer_size_ms = 30 * 1000; // 30 seconds

/*
* Port of the mqtt server to connect to; Default: 1883
*/
static inline constexpr uint32_t mqtt_server_port = 1883;

/*
* IP address of the mqtt server
*/
const std::string mqtt_server_ip = "";

/*
* MQTT Topic to publish/update status on
*/
const std::string mqtt_topic = "status/server";

/*
* Port of the RTP receiver
*/
const int rtp_recv_port = 5004;

/* 
* Settings for the model inference, 
* Such as usage of gpu, threads etc.
*/
struct model_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t n_gpu_layers = 999;

    float vad_thold  = 0.6f;
    float freq_thold = 100.0f;

    bool speed_up       = false;
    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;
    bool no_timestamps  = true;
    bool verbose_prompt = false;
    bool use_gpu        = true;
    bool flash_attn     = false;

    std::string person      = "Georgi";
    std::string bot_name    = "LLaMA";
    std::string wake_cmd    = "";
    std::string heard_ok    = "";
    std::string language    = "nl";
    std::string model_wsp   = "../models/ggml-medium.bin";
    std::string model_llama = "../models/GEITje-7B-chat-v2.gguf";
    std::string speak       = "./../speak";
    std::string speak_file  = "./../to_speak";
    std::string prompt      = "";
    std::string fname_out;
    std::string path_session = "";       // path to file for saving/loading model eval state
};

#endif /* SETTINGS_HPP */