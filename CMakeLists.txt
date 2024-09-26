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

set(WHISPER_CUDA ON)

set(LLAMA_CUDA ON)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp
    GIT_TAG v1.6.2
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/whisper
)
FetchContent_MakeAvailable(whisper)

add_definitions("-DGGML_USE_CUDA")

FetchContent_MakeAvailable(openwakeword)

find_package(OpenCV REQUIRED)

add_subdirectory(src/person_detector)
add_subdirectory(nervousPico_connectivity)

add_library(llama ${CMAKE_CURRENT_LIST_DIR}/lib/llama/llama.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/unicode.cpp
                  ${CMAKE_CURRENT_LIST_DIR}/lib/llama/unicode-data.cpp)
target_include_directories(llama PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/llama/)
target_link_libraries(llama whisper)

add_library(common ${CMAKE_CURRENT_LIST_DIR}/src/common/common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/common)

add_library(whisper_wrapper ${CMAKE_CURRENT_LIST_DIR}/src/whisper_wrapper/whisper_wrapper.cpp)
target_include_directories(whisper_wrapper PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/whisper_wrapper ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(whisper_wrapper whisper common)

add_library(llama_wrapper ${CMAKE_CURRENT_LIST_DIR}/src/llama_wrapper/llama_wrapper.cpp)
target_include_directories(llama_wrapper PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/llama_wrapper ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(llama_wrapper whisper whisper_wrapper llama common)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)

include_directories(${MOSQUITTO_INCLUDE_DIRS})
link_directories(${MOSQUITTO_LIBRARY_DIRS})

add_executable(
        robo_app
        src/main.cpp)

target_include_directories(robo_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(robo_app
        ${MOSQUITTO_LIBRARIES}
        ${OpenCV_LIBS}
        openwakeword 
        person_detector
        llama_wrapper
        whisper_wrapper
        CLI_HOST
)

