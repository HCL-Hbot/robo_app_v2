#include <whisper_wrapper.hpp>
#include <whisper_helper.hpp>
#include <common.h>
#include <regex>

whisper_wrapper::whisper_wrapper(model_params params)
{
    m_params = params;
    m_cparams.use_gpu = params.use_gpu;
    m_cparams.flash_attn = params.flash_attn;
    m_ctx = whisper_init_from_file_with_params(m_params.model_wsp.c_str(), m_cparams);
    if (!m_ctx)
    {
        fprintf(stderr, "No whisper.cpp model specified. Please provide using -mw <modelfile>\n");
        exit(1);
    }
}
void whisper_wrapper::init(bool force_speak)
{
    m_force_speak = force_speak;
    m_prompt_whisper = ::replace(k_prompt_whisper, "{1}", m_params.bot_name);
}

std::string whisper_wrapper::do_inference(std::vector<float> &pcmf_samples)
{
    std::string all_heard;
    if (!m_force_speak)
    {
        all_heard = ::trim(::transcribe(m_ctx, m_params, pcmf_samples, m_prompt_whisper, m_prob0, m_t_ms));
    }

    const auto words = get_words(all_heard);

    std::string wake_cmd_heard;
    std::string text_heard;

    for (int i = 0; i < (int)words.size(); ++i)
    {
        text_heard += words[i] + " ";
    }
    
    // remove text between brackets using regex
    {
        std::regex re("\\[.*?\\]");
        text_heard = std::regex_replace(text_heard, re, "");
    }

    // remove text between brackets using regex
    {
        std::regex re("\\(.*?\\)");
        text_heard = std::regex_replace(text_heard, re, "");
    }

    // remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '
    text_heard = std::regex_replace(text_heard, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), "");

    // take first line
    text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));

    // remove leading and trailing whitespace
    text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
    text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");
    return text_heard;
}