cmake_minimum_required(VERSION 3.8)
project(iterative_closest_point)


find_package(PCL 1.5 REQUIRED)


include_directories(
    include
    ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(executable_name icp_test)
set(library_name icp)

add_library(${library_name}
                src/iterative_closest_point.cpp
            )

add_executable(${executable_name}
                src/main.cpp
)

target_link_libraries(${executable_name} ${library_name} ${PCL_LIBRARIES})