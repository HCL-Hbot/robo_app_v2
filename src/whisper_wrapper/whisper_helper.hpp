#ifndef WHISPER_HELPER_HPP
#define WHISPER_HELPER_HPP
#include <cstdint>
#include <string>
#include <whisper.h>
#include <chrono>
#include <thread>
#include <vector>
#include <sstream>
#include <settings.hpp>

/**
 * @brief Transcribe audio data using a Whisper model.
 * 
 * This function transcribes the provided PCM audio data using the specified Whisper model context
 * and parameters. It returns the transcription result as a string.
 * 
 * @param ctx Pointer to the Whisper model context.
 * @param params The parameters for the model.
 * @param pcmf32 The audio data to transcribe, as a vector of 32-bit floats.
 * @param prompt_text The initial prompt text to use for transcription.
 * @param prob Reference to a float to store the average probability of the transcription tokens.
 * @param t_ms Reference to an int64_t to store the transcription time in milliseconds.
 * @return std::string The transcription result.
 */
std::string transcribe(
        whisper_context * ctx,
        const model_params & params,
        const std::vector<float> & pcmf32,
        const std::string prompt_text,
        float & prob,
        int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    std::vector<whisper_token> prompt_tokens;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    prompt_tokens.resize(1024);
    prompt_tokens.resize(whisper_tokenize(ctx, prompt_text.c_str(), prompt_tokens.data(), prompt_tokens.size()));

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.prompt_tokens    = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
    wparams.prompt_n_tokens  = prompt_tokens.empty() ? 0       : prompt_tokens.size();

    wparams.audio_ctx        = params.audio_ctx;
    wparams.speed_up         = params.speed_up;

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}

/**
 * @brief Split a string into words.
 * 
 * This function splits the given text into words, using whitespace as the delimiter,
 * and returns them as a vector of strings.
 * 
 * @param txt The input text to split into words.
 * @return std::vector<std::string> A vector containing the words from the input text.
 */
std::vector<std::string> get_words(const std::string &txt) {
    std::vector<std::string> words;

    std::istringstream iss(txt);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

/* Prompt to prepare whisper for current talking session */
const std::string k_prompt_whisper = R"(A conversation with a person called {1}.)";

#endif /* WHISPER_HELPER_HPP */