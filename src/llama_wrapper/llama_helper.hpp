#ifndef LLAMA_HELPER_HPP
#define LLAMA_HELPER_HPP
#include <llama.h>
#include <string>
#include <algorithm>

//const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
//{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}’s requests immediately and with details and precision.
//There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
//The transcript only includes text, it does not include markup like HTML and Markdown.
//{1} responds with short and concise answers.

//{0}{4} Hello, {1}!
//{1}{4} Hello {0}! How may I help you today?
//{0}{4} What time is it?
//{1}{4} It is {2} o'clock.
//{0}{4} What year is it?
//{1}{4} We are in {3}.
//{0}{4} What is a cat?
//{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
//{0}{4} Name a color.
//{1}{4} Blue
//{0}{4})";


const std::string k_prompt_llama = R"(Teksttranscript van een eindeloze dialoog, waarin {0} interacteert met een AI-assistent genaamd {1}.
{1} is behulpzaam, vriendelijk, eerlijk, aardig, goed in schrijven en slaagt er altijd in om de verzoeken van {0} onmiddellijk en met details en precisie te beantwoorden.
Er zijn geen annotaties zoals (30 seconden later...) of (tegen zichzelf), alleen wat {0} en {1} hardop tegen elkaar zeggen.
Het transcript bevat alleen tekst, geen opmaak zoals HTML of Markdown.
{1} geeft korte en bondige antwoorden.

{0} is een wat oudere persoon, ongeveer 80 jaar oud.

Alle antwoorden moeten in het Nederlands zijn.

{0}{4} Hallo, {1}!
{1}{4} Hallo {0}! Hoe kan ik u vandaag helpen?
{0}{4} Hoe laat is het?
{1}{4} Het is {2} uur.
{0}{4} Welk jaar is het?
{1}{4} We zijn in het jaar {3}.
{0}{4} Wat is een kat?
{1}{4} Een kat is een gedomesticeerde soort van kleine vleesetende zoogdieren. Het is de enige gedomesticeerde soort in de familie van de Felidae.
{0}{4} Noem een kleur.
{1}{4} Blauw
{0}{4})";


std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    auto * model = llama_get_model(ctx);

    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

static std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::string construct_initial_prompt_for_llama(model_params &params){
    const std::string chat_symb = ":";
// construct the initial prompt for LLaMA inference
    std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;

    // need to have leading ' '
    prompt_llama.insert(0, 1, ' ');

    prompt_llama = ::replace(prompt_llama, "{0}", params.person);
    prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);

    {
        // get time string
        std::string time_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%H:%M", now);
            time_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{2}", time_str);
    }

    {
        // get year string
        std::string year_str;
        {
            time_t t = time(0);
            struct tm * now = localtime(&t);
            char buf[128];
            strftime(buf, sizeof(buf), "%Y", now);
            year_str = buf;
        }
        prompt_llama = ::replace(prompt_llama, "{3}", year_str);
    }

    prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);
    return prompt_llama;
}
#endif /* LLAMA_HELPER_HPP */
