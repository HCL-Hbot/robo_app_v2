find_package(Boost REQUIRED)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/json
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(person_detector person_detector.cpp)
target_include_directories(person_detector PUBLIC "." ${BOOST_INCLUDE_DIRS})
target_link_libraries(person_detector CLI_HOST ${BOOST_LIBRARIES} nlohmann_json::nlohmann_json)