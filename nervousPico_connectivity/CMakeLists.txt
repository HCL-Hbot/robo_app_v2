FetchContent_Declare(
  libserialport
  GIT_REPOSITORY git://sigrok.org/libserialport
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/libserialport
)

FetchContent_MakeAvailable(libserialport)

if(UNIX AND NOT APPLE)
    add_library(libserialport 
      libserialport/linux.c 
      libserialport/linux_termios.c 
      libserialport/serialport.c 
      libserialport/timing.c
    )
    target_compile_definitions(libserialport PUBLIC "-DSP_PRIV=__attribute__((visibility(\"hidden\")))")
elseif(WIN32)
    add_library(libserialport 
      libserialport/windows.c
    )
else()
    add_library(libserialport 
      libserialport/macosx.c
    )
endif()

target_include_directories(libserialport PUBLIC libserialport/ ${CMAKE_CURRENT_LIST_DIR})

# Define CLI_HOST as a header-only library
add_library(CLI_HOST INTERFACE)

target_include_directories(CLI_HOST INTERFACE "${CMAKE_CURRENT_LIST_DIR}")

target_link_libraries(CLI_HOST INTERFACE libserialport)