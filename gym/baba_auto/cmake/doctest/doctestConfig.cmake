# Stands in for doctest's own package config. baba-is-auto's top-level
# CMakeLists requires doctest for its unit tests, which this package never
# builds (setup.py builds the pyBaba target alone), so an empty interface
# target is enough for the test sources to be declared without the header.
if(NOT TARGET doctest::doctest)
    add_library(doctest::doctest INTERFACE IMPORTED)
endif()
