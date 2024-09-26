    #include <llama_wrapper.hpp>
    

    llama_wrapper::llama_wrapper(model_params params)
    {
        m_params = params;
        llama_backend_init();
        if (!params.use_gpu)
        {
            m_lmparams.n_gpu_layers = 0;
        }
        else
        {
            m_lmparams.n_gpu_layers = params.n_gpu_layers;
        }

        model_llama = llama_load_model_from_file(params.model_llama.c_str(), m_lmparams);
        if (!model_llama)
        {
            fprintf(stderr, "No llama.cpp model specified. Please provide using -ml <modelfile>\n");
            exit(1);
        }

        // tune these to your liking
        m_lcparams.n_ctx = 2048;
        m_lcparams.seed = 1;
        m_lcparams.n_threads = params.n_threads;
        m_lcparams.flash_attn = params.flash_attn;

        m_ctx = llama_new_context_with_model(model_llama, m_lcparams);
    }

    void llama_wrapper::init()
    {

        std::string prompt_llama = construct_initial_prompt_for_llama(m_params);

        m_batch = llama_batch_init(llama_n_ctx(m_ctx), 0, 1);

        m_path_session = m_params.path_session;

        m_embd_inp = ::llama_tokenize(m_ctx, prompt_llama, true);

        if (!m_path_session.empty())
        {
            fprintf(stderr, "%s: attempting to load saved session from %s\n", __func__, m_path_session.c_str());

            // fopen to check for existing session
            FILE *fp = std::fopen(m_path_session.c_str(), "rb");
            if (fp != NULL)
            {
                std::fclose(fp);

                m_session_tokens.resize(llama_n_ctx(m_ctx));
                size_t n_token_count_out = 0;

                if (!llama_state_load_file(m_ctx, m_path_session.c_str(), m_session_tokens.data(), m_session_tokens.capacity(), &n_token_count_out))
                {
                    fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, m_path_session.c_str());
                    exit(1);
                }
                m_session_tokens.resize(n_token_count_out);
                for (size_t i = 0; i < m_session_tokens.size(); i++)
                {
                    m_embd_inp[i] = m_session_tokens[i];
                }

                fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int)m_session_tokens.size());
            }
            else
            {
                fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
            }
        }

        prepare_batch(m_embd_inp, 0);

        if (llama_decode(m_ctx, m_batch))
        {
            fprintf(stderr, "%s : failed to decode\n", __func__);
            exit(1);
        }

        if (m_params.verbose_prompt)
        {
            fprintf(stdout, "\n");
            fprintf(stdout, "%s", prompt_llama.c_str());
            fflush(stdout);
        }

        // debug message about similarity of saved session, if applicable
        size_t n_matching_session_tokens = 0;
        if (m_session_tokens.size())
        {
            for (llama_token id : m_session_tokens)
            {
                if (n_matching_session_tokens >= m_embd_inp.size() || id != m_embd_inp[n_matching_session_tokens])
                {
                    break;
                }
                n_matching_session_tokens++;
            }
            if (n_matching_session_tokens >= m_embd_inp.size())
            {
                fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
            }
            else if (n_matching_session_tokens < (m_embd_inp.size() / 2))
            {
                fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                        __func__, n_matching_session_tokens, m_embd_inp.size());
            }
            else
            {
                fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                        __func__, n_matching_session_tokens, m_embd_inp.size());
            }
        }

        // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
        // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
        // initial prompt so it doesn't need to be an exact match.
        m_need_to_save_session = !m_path_session.empty() && n_matching_session_tokens < (m_embd_inp.size() * 3 / 4);

        printf("%s : done! start speaking in the microphone\n", __func__);

        printf("\n");
        // printf("%s%s", m_params.person.c_str(), m_chat_symb.c_str());
        fflush(stdout);

        // text inference variables
        m_n_keep = m_embd_inp.size();
        m_n_ctx = llama_n_ctx(m_ctx);

        m_n_past = m_n_keep;
        m_n_prev = 64; // TODO arg
        m_n_session_consumed = !m_path_session.empty() && m_session_tokens.size() > 0 ? m_session_tokens.size() : 0;

        // reverse prompts for detecting when it's time to stop speaking
        m_antiprompts = {
            m_params.person + ":",
        };
    }

    std::string llama_wrapper::do_inference(std::string text_heard)
    {
        const std::vector<llama_token> tokens = llama_tokenize(m_ctx, text_heard.c_str(), false);

        if (tokens.empty())
        {
            // fprintf(stdout, "%s: Heard nothing, skipping ...\n", __func__);

            return "";
        }

        text_heard.insert(0, 1, ' ');
        text_heard += "\n" + m_params.bot_name + ":";

        m_embd = ::llama_tokenize(m_ctx, text_heard, false);
        // Append the new input tokens to the session_tokens vector
        if (!m_path_session.empty())
        {
            m_session_tokens.insert(m_session_tokens.end(), tokens.begin(), tokens.end());
        }

        // text inference
        bool done = false;
        std::string text_to_speak;
        while (true)
        {
            // predict
            if (m_embd.size() > 0)
            {
                if (m_n_past + (int)m_embd.size() > m_n_ctx)
                {
                    m_n_past = m_n_keep;

                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    m_embd.insert(m_embd.begin(), m_embd_inp.begin() + m_embd_inp.size() - m_n_prev, m_embd_inp.end());
                    // stop saving session if we run out of context
                    m_path_session = "";
                }

                // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                // REVIEW
                if (m_n_session_consumed < (int)m_session_tokens.size())
                {
                    size_t i = 0;
                    for (; i < m_embd.size(); i++)
                    {
                        if (m_embd[i] != m_session_tokens[m_n_session_consumed])
                        {
                            m_session_tokens.resize(m_n_session_consumed);
                            break;
                        }

                        m_n_past++;
                        m_n_session_consumed++;

                        if (m_n_session_consumed >= (int)m_session_tokens.size())
                        {
                            i++;
                            break;
                        }
                    }
                    if (i > 0)
                    {
                        m_embd.erase(m_embd.begin(), m_embd.begin() + i);
                    }
                }

                if (m_embd.size() > 0 && !m_path_session.empty())
                {
                    m_session_tokens.insert(m_session_tokens.end(), m_embd.begin(), m_embd.end());
                    m_n_session_consumed = m_session_tokens.size();
                }

                prepare_batch(m_embd, m_n_past);

                if (llama_decode(m_ctx, m_batch))
                {
                    fprintf(stderr, "%s : failed to decode\n", __func__);
                    exit(1);
                }
            }

            m_embd_inp.insert(m_embd_inp.end(), m_embd.begin(), m_embd.end());
            m_n_past += m_embd.size();

            m_embd.clear();

            if (done)
                break;

            {
                // out of user input, sample next token
                const float top_k = 5;
                const float top_p = 0.80f;
                const float temp = 0.30f;
                const float repeat_penalty = 1.1764f;

                const int repeat_last_n = 256;

                if (!m_path_session.empty() && m_need_to_save_session)
                {
                    m_need_to_save_session = false;
                    llama_state_save_file(m_ctx, m_path_session.c_str(), m_session_tokens.data(), m_session_tokens.size());
                    // llama_save_session_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size());
                }

                llama_token id = 0;

                {
                    auto logits = llama_get_logits(m_ctx);
                    auto n_vocab = llama_n_vocab(model_llama);

                    logits[llama_token_eos(model_llama)] = 0;

                    std::vector<llama_token_data> candidates;
                    candidates.reserve(n_vocab);
                    for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                    {
                        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                    }

                    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

                    // apply repeat penalty
                    const float nl_logit = logits[llama_token_nl(model_llama)];

                    llama_sample_repetition_penalties(m_ctx, &candidates_p,
                                                      m_embd_inp.data() + std::max(0, m_n_past - repeat_last_n),
                                                      repeat_last_n, repeat_penalty, 0.0, 0.0f);

                    logits[llama_token_nl(model_llama)] = nl_logit;

                    if (temp <= 0)
                    {
                        // Greedy sampling
                        id = llama_sample_token_greedy(m_ctx, &candidates_p);
                    }
                    else
                    {
                        // Temperature sampling
                        llama_sample_top_k(m_ctx, &candidates_p, top_k, 1);
                        llama_sample_top_p(m_ctx, &candidates_p, top_p, 1);
                        llama_sample_temp(m_ctx, &candidates_p, temp);
                        id = llama_sample_token(m_ctx, &candidates_p);
                    }
                }

                if (id != llama_token_eos(model_llama))
                {
                    // add it to the context
                    m_embd.push_back(id);

                    text_to_speak += llama_token_to_piece(m_ctx, id);

                    printf("%s", llama_token_to_piece(m_ctx, id).c_str());
                    fflush(stdout);
                }
            }

            {
                std::string last_output;
                for (int i = m_embd_inp.size() - 16; i < (int)m_embd_inp.size(); i++)
                {
                    last_output += llama_token_to_piece(m_ctx, m_embd_inp[i]);
                }
                last_output += llama_token_to_piece(m_ctx, m_embd[0]);

                for (std::string &antiprompt : m_antiprompts)
                {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
                    {
                        done = true;
                        text_to_speak = ::replace(text_to_speak, antiprompt, "");
                        fflush(stdout);
                        m_need_to_save_session = true;
                        break;
                    }
                }
            }

            m_is_running = sdl_poll_events();

            if (!m_is_running)
            {
                break;
            }
        }
        return text_to_speak;
    }