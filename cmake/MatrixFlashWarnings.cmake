# Warning policy for MatrixFlash-Pro.
#
# Design decision: warnings are advisory, never fatal. The library predates this
# policy and must keep building unchanged, so /WX (or -Werror) is intentionally
# NOT applied -- the CI-style check is "read the warning count", not "break the
# build".

include_guard(GLOBAL)

# matrix_pro_set_warnings(<target>)
#   Applies the project warning set to the C++ sources of <target>.
#   NVCC-compiled sources are left untouched unless MATRIX_PRO_CUDA_WARNINGS=ON,
#   because forwarding /W4 into nvcc produces a large amount of third-party
#   header noise that hides genuine findings in our own kernels.
function(matrix_pro_set_warnings target)
  if(NOT MATRIX_PRO_WARNINGS)
    return()
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/permissive->
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
    if(MATRIX_PRO_CUDA_WARNINGS)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
    if(MATRIX_PRO_CUDA_WARNINGS)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
    endif()
  endif()
endfunction()

# Helper for the many small executables (tests/examples/benchmarks) that all
# need the same treatment: warnings + a sane C++ standard.
function(matrix_pro_add_warnings_to_all)
  if(NOT ARGN)
    return()
  endif()
  foreach(target IN LISTS ARGN)
    if(TARGET ${target})
      matrix_pro_set_warnings(${target})
    endif()
  endforeach()
endfunction()