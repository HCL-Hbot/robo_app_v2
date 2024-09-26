#ifndef LLAMA_WRAPPER_HPP
#define LLAMA_WRAPPER_HPP
#include <whisper_helper.hpp>
#include <llama.h>
#include <common.h>
#include "llama_helper.hpp"
#include <string>
#include <settings.hpp>

class llama_wrapper
{
public:
    /**
     * @brief Construct a new llama wrapper object
     *
     * @param params A struct containing configuration for model inference; Runtime as well as behavioural settings
     */
    llama_wrapper(model_params params);

    /**
     * @brief Initialize llama by initializing a talk session with a initializer response
     */
    void init();

    /**
     * @brief Run inference (this reuses the initialised session)
     * 
     * @param text_heard user_prompt, to react on
     * @return std::string The response from the llama model
     */
    std::string do_inference(std::string text_heard);

    /**
     * @brief Destroy the llama wrapper object
     * 
     */
    ~llama_wrapper()
    {
        llama_print_timings(m_ctx);
        llama_free(m_ctx);
    }

private:
    /* This is used to store model details (tokens, architecture, etc) about the to be loaded model */
    llama_model *model_llama;

    /* This is the context that is used with the model and session
    * When used with the session tokens, it can make the model react on previous responses (learn from it, kinda) 
    */
    llama_context *m_ctx;

    /* Configuration parameters for the model and context */
    /* The default settings should be fine, and give a good starting ground */
    llama_context_params m_lcparams = llama_context_default_params();
    llama_model_params m_lmparams = llama_model_default_params();

    /* A copy of the parameters passed to the constructor */
    /* The settings are used multiple times during inference and configuration */
    model_params m_params;

    /* Session tokens which are used to help do inference on text prompts */
    std::vector<llama_token> m_session_tokens;

    /* Antiprompts are used to prevent the model from going mayhem when prompt is empty or invalid text is put in to the prompt */
    std::vector<std::string> m_antiprompts;

    /* Tokens of prompt that gets infered */
    std::vector<llama_token> m_embd;
    std::vector<gpt_vocab::id> m_embd_inp;

    /* Batch, this is used for decoding the response of the model*/
    llama_batch m_batch;

    /* Sessions can be loaded/unloaded to speed up initializing of the model for inference */
    /* This variable contains the path to the file where last session was saved to */
    std::string m_path_session;
    bool m_need_to_save_session;

    int m_n_past;
    int m_n_prev;
    
    int m_n_session_consumed;
    
    const int m_voice_id = 2;
    
    int m_n_keep;
    
    int m_n_ctx;
    
    bool m_is_running;

    /* Prepare the batch, so that it can be used later for decoding responses */
    void prepare_batch(std::vector<llama_token> &tokens, int n_past)
    {
        {
            m_batch.n_tokens = tokens.size();

            for (int i = 0; i < m_batch.n_tokens; i++)
            {
                m_batch.token[i] = tokens[i];
                m_batch.pos[i] = n_past + i;
                m_batch.n_seq_id[i] = 1;
                m_batch.seq_id[i][0] = 0;
                m_batch.logits[i] = i == m_batch.n_tokens - 1;
            }
        }
    }
};

#endif /* LLAMA_WRAPPER_HPP */