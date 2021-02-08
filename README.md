# cmsregr-oneapi
cms regression with oneapi

## Description
Standalone simplified CMS Ecal Local Reco using standard c++ and Intel OneAPI.

## Requirements
 - the easiest is to have `cvmfs`
 - `gcc` stuff was taken from `cvmfs`
 - `root` libs/includes
 - `oneapi` was taken from `cvmfs` as well
 - `Eigen` was taken from `gitlab` master and patched to be able to compile for device (cpu in this case) (not pushed anywhere yet, but updates were similar to what was done for enabling Eigen be part of `cuda` kernels)

## Results
 - `multifit_cpp` contains plain c++ impl
 - `multifit_oneapi_dpct` is the `oneapi` one
   - `dpcpp` compatibility tool was used initially on a `cuda` based impl
   - `Eigen` was cloned and updated in place, notable updates are
      - removing the inlined assembly instructions (SPIRV capability 5606)
      - removing dynamic stack allocations (SPIRV capability 5817)
      - the previous one shows up as `VariableLengthArrays`... which is sort of similar to dynamic stack allocation, kinda
 - both plain c++ and `oneapi` impls produce identical results when compared to the trith values, known from apriori generation...
