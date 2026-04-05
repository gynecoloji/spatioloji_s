pybind11_add_module(_scoring_cpp scoring_cpp.cpp)

# ── Optional OpenMP support ───────────────────────────────────────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_scoring_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_scoring_cpp PRIVATE SPATIOLOJI_USE_OPENMP)
endif()

install(TARGETS _scoring_cpp LIBRARY DESTINATION spatioloji_s/ccc)
