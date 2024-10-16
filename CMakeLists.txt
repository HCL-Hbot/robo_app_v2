cmake_minimum_required(VERSION 3.6)
project(robo_app)

include(FetchContent)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

FetchContent_Declare(
        openwakeword
        GIT_REPOSITORY https://github.com/Hoog-V/Openwakeword.Cpp.git
        GIT_TAG main
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/Openwakeword.cpp
)


SET(GGML_CUDA ON)

# set(WHISPER_CUDA ON)

set(LLAMA_CUDA ON)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp
    GIT_TAG master
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/whisper
)

# Used Models
FetchContent_Declare(
     whisper_model
     URL https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin
     SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/model
     DOWNLOAD_NO_EXTRACT 1
)

FetchContent_Declare(
     llama_model
     URL https://huggingface.co/BramVanroy/fietje-2-chat-gguf/resolve/main/fietje-2b-chat-q6_k.gguf
     SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/model
     DOWNLOAD_NO_EXTRACT 1
)

FetchContent_Declare(
     piper_model
     URL https://huggingface.co/rhasspy/piper-voices/resolve/main/nl/nl_BE/nathalie/medium/nl_BE-nathalie-medium.onnx
     SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/model
     DOWNLOAD_NO_EXTRACT 1
)

FetchContent_Declare(
     piper_model_config
     URL https://huggingface.co/rhasspy/piper-voices/resolve/main/nl/nl_BE/nathalie/medium/nl_BE-nathalie-medium.onnx.json
     SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/model
     DOWNLOAD_NO_EXTRACT 1
)

FetchContent_Declare(
     piper
     URL https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
     SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/piper
)
FetchContent_MakeAvailable(whisper_model)
FetchContent_MakeAvailable(llama_model)
FetchContent_MakeAvailable(piper_model)
FetchContent_MakeAvailable(piper_model_config)
FetchContent_MakeAvailable(piper)


FetchContent_MakeAvailable(whisper)

FetchContent_MakeAvailable(openwakeword)

find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_subdirectory(src/person_detector)
add_subdirectory(nervousPico_connectivity)

add_library(llama ${CMAKE_CURRENT_LIST_DIR}/lib/llama/llama.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/llama-grammar.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/llama-sampling.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/llama-vocab.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/unicode.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/unicode-data.cpp)
target_include_directories(llama PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/llama/)
target_link_libraries(llama PUBLIC whisper ggml)

add_library(common ${CMAKE_CURRENT_LIST_DIR}/src/common/common.cpp ${CMAKE_CURRENT_LIST_DIR}/src/common/common-sdl.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/common ${SDL2_INCLUDE_DIRS})
target_link_libraries(common ${SDL2_LIBS})

add_library(whisper_wrapper ${CMAKE_CURRENT_LIST_DIR}/src/whisper_wrapper/whisper_wrapper.cpp)
target_include_directories(whisper_wrapper PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/whisper_wrapper ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(whisper_wrapper whisper common)

add_library(llama_wrapper ${CMAKE_CURRENT_LIST_DIR}/src/llama_wrapper/llama_wrapper.cpp)
target_include_directories(llama_wrapper PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/llama_wrapper ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(llama_wrapper PUBLIC whisper whisper_wrapper llama common)

add_executable(
        robo_app
        src/main.cpp)

target_include_directories(robo_app PRIVATE ${OpenCV_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS})
target_link_libraries(robo_app
        ${OpenCV_LIBS}
        ${SDL2_LIBS}
        openwakeword 
        person_detector
        llama_wrapper
        whisper_wrapper
        llama 
        whisper
        CLI_HOST
)

