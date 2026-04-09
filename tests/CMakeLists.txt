project(tests LANGUAGES CXX)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(TARGET_NAME "tensor-test")
add_executable(${TARGET_NAME} tests.cpp)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(${TARGET_NAME}
    ${GTEST_LIBRARIES}
)

target_compile_options(${TARGET_NAME} PUBLIC ${COMPILE_OPT})
target_link_options(${TARGET_NAME} PUBLIC ${LINK_OPT})
