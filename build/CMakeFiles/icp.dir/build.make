# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tony/iterative_closest_point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tony/iterative_closest_point/build

# Include any dependencies generated for this target.
include CMakeFiles/icp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/icp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/icp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/icp.dir/flags.make

CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o: CMakeFiles/icp.dir/flags.make
CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o: ../src/iterative_closest_point.cpp
CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o: CMakeFiles/icp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tony/iterative_closest_point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o -MF CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o.d -o CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o -c /home/tony/iterative_closest_point/src/iterative_closest_point.cpp

CMakeFiles/icp.dir/src/iterative_closest_point.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/icp.dir/src/iterative_closest_point.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tony/iterative_closest_point/src/iterative_closest_point.cpp > CMakeFiles/icp.dir/src/iterative_closest_point.cpp.i

CMakeFiles/icp.dir/src/iterative_closest_point.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/icp.dir/src/iterative_closest_point.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tony/iterative_closest_point/src/iterative_closest_point.cpp -o CMakeFiles/icp.dir/src/iterative_closest_point.cpp.s

# Object files for target icp
icp_OBJECTS = \
"CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o"

# External object files for target icp
icp_EXTERNAL_OBJECTS =

libicp.a: CMakeFiles/icp.dir/src/iterative_closest_point.cpp.o
libicp.a: CMakeFiles/icp.dir/build.make
libicp.a: CMakeFiles/icp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tony/iterative_closest_point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libicp.a"
	$(CMAKE_COMMAND) -P CMakeFiles/icp.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/icp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/icp.dir/build: libicp.a
.PHONY : CMakeFiles/icp.dir/build

CMakeFiles/icp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/icp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/icp.dir/clean

CMakeFiles/icp.dir/depend:
	cd /home/tony/iterative_closest_point/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tony/iterative_closest_point /home/tony/iterative_closest_point /home/tony/iterative_closest_point/build /home/tony/iterative_closest_point/build /home/tony/iterative_closest_point/build/CMakeFiles/icp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/icp.dir/depend

