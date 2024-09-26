#ifndef WHISPER_WRAPPER_HPP
#define WHISPER_WRAPPER_HPP
#include <whisper.h>
#include <whisper_helper.hpp>
#include <string>

/**
 * @class whisper_wrapper
 * @brief A class to handle Whisper model operations.
 * 
 * This class initializes the Whisper model, performs inference on audio data, and manages
 * related parameters and context.
 */
class whisper_wrapper
{
public:
    /**
     * @brief Construct a new whisper_wrapper object.
     * 
     * @param params The parameters for the Whisper model.
     */
    whisper_wrapper(model_params params);

    /**
     * @brief Initialize the Whisper model.
     * 
     * @param force_speak Flag to force speaking mode.
     */
    void init(bool force_speak = false);

    /**
     * @brief Perform inference on the provided audio samples.
     * 
     * This method transcribes the provided PCM audio samples using the Whisper model.
     * 
     * @param pcmf_samples The audio samples to transcribe.
     * @return std::string The transcription result.
     */
    std::string do_inference(std::vector<float> &pcmf_samples);

    /**
     * @brief Destroy the whisper_wrapper object.
     * 
     * This destructor prints the timings and frees the Whisper context.
     */
    ~whisper_wrapper()
    {
        whisper_print_timings(m_ctx);
        whisper_free(m_ctx);
    }

private:
    int64_t m_t_ms = 0;                  ///< Inference timing in milliseconds.
    whisper_context *m_ctx;              ///< Whisper model context.
    model_params m_params;               ///< Parameters for the Whisper model.
    std::vector<float> m_pcmf32_prompt;  ///< PCM audio samples for the prompt.
    std::string m_prompt_whisper;        ///< Prompt text for the Whisper model.
    const std::string m_chat_symb = ":"; ///< Chat symbol used in the prompt.
    float m_prob0 = 0.0f;                ///< Initial probability.
    bool m_force_speak;                  ///< Flag to force speaking mode.
    struct whisper_context_params m_cparams = whisper_context_default_params(); ///< Default context parameters.
};

#endif /* WHISPER_WRAPPER_HPP */