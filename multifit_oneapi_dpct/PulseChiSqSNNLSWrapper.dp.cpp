#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "PulseChiSqSNNLSWrapper.h"
#include "PulseChiSqSNNLS.h"

#include <vector>
#include <iostream>
#include <string>

using namespace sycl;

void show_platforms() {
        auto platforms = platform::get_platforms();

            for (auto& p : platforms) {
                        std::cout << "Platform: "
                                              << p.get_info<info::platform::name>()
                                                                << std::endl;

                                auto devs = p.get_devices();
                                        for (auto& d : devs)
                                                        std::cout << "  Device: "
                                                                                  << d.get_info<info::device::name>()
                                                                                                        << std::endl;
                                            }   
}


constexpr int NValues = 10000;

void initArray(std::vector<int>& arr) {
    for (int i=0; i<arr.size(); i++)
        arr[i] = i;
}

int test1() {
    // queue
    queue q;

    // data on the host
    std::vector<int> AHost(NValues), BHost(NValues), CHost(NValues), CHostTest(NValues);
    initArray(AHost); initArray(BHost);
    for (int i=0; i<AHost.size(); i++)
        CHostTest[i] = AHost[i] + BHost[i];

    {
        // buffer objs
        buffer<int, 1> A{AHost};
        buffer<int, 1> B{BHost};
        buffer<int, 1> C{CHost.data(), CHost.size()};
        C.set_write_back(false);

        // out cmd group
        q.submit([&](handler &h){
            // prerequisites
            auto Adev = A.get_access<access::mode::read>(h);
            auto Bdev = B.get_access<access::mode::read>(h);
            auto Cdev = C.get_access<access::mode::write>(h);

            h.parallel_for(
                nd_range<1>{range<1>{NValues}, range<1>{1}},
                [=](nd_item<1> item) {
                    int i = item.get_global_id(0);
                    Cdev[i] = Adev[i] + Bdev[i];
                }
            );
        });
    }

    for (int i=0; i<CHost.size(); i++)
        assert(CHost[i] == CHostTest[i] && "Vector Addition failed on device");

    std::cout << "Vector Addition succeeded\n";
    for (int i=0; i<CHost.size(); i++)
        if (i<3 or i==CHost.size()-1)
            std::cout << "C[" << i << "] = " << CHost[i] << std::endl;

    return 0;
}

std::vector<Output> doFitWrapper(std::vector<DoFitArgs> const &vargs) {
    show_platforms();
    //dpct::device_ext &dev_ct1 = dpct::get_current_device();
    //sycl::queue &q_ct1 = dev_ct1.default_queue();
    queue q_ct1;
    auto dev = q_ct1.get_device();
    std::cout << "will use device: \n";
    std::cout << dev.get_info<info::device::name>() << std::endl;
    // input parameters to the multifit on gpu
    DoFitArgs* d_args;
    Output* d_results;
    std::vector<Output> results;
    std::cout << "vargs.size() = " << vargs.size() << std::endl;
    results.resize(vargs.size());
    std::cout << "size = " << results.size() << std::endl;
    std::cout << "capacity = " << results.capacity() << std::endl;

    // allocate on the device
    std::cout << "allocate on the device" << std::endl;
    d_args = sycl::malloc_device<DoFitArgs>(vargs.size(), q_ct1);
    d_results = sycl::malloc_device<Output>(vargs.size(), q_ct1);

    // transfer to the device
    std::cout << "copy to the device " << std::endl;
    q_ct1.memcpy(d_args, vargs.data(), sizeof(DoFitArgs) * vargs.size()).wait();

    // kernel invoacation
    std::cout << "launch the kenrel" << std::endl;
    //int nthreadsPerBlock = 256;
    //int nblocks = (vargs.size() + nthreadsPerBlock - 1) / nthreadsPerBlock;
    unsigned int nthreadsPerBlock = 4;
    
    //DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    //limit. To get the device limit, query info::device::max_work_group_size.
    //Adjust the workgroup size if needed.
    auto start_time = std::chrono::high_resolution_clock::now();
    q_ct1.submit([&](sycl::handler &cgh) {
        auto vargs_size_ct2 = vargs.size();

        stream out{1024, 680, cgh};
        cgh.parallel_for(
            /*
            nd_range<1>{range<1>{100}, range<1>{1}},
            [=](nd_item<1> item) {
                out << item.get_global_id(0) << "  " << item.get_local_id(0) << "\n";
            }
            */

            sycl::nd_range<1>{sycl::range<1>{vargs.size()}, sycl::range<1>{nthreadsPerBlock}},
            [=](sycl::nd_item<1> item_ct1) {
                //out << item_ct1.get_global_id(0) << " " << item_ct1.get_local_id(0) << "\n";

                kernel_multifit(d_args, d_results, vargs_size_ct2, item_ct1);
            }
        );
    });
    //dev_ct1.queues_wait_and_throw();
    q_ct1.wait();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    std::cout << "kernel only duration = " << duration << std::endl;

    // copy results back
    std::cout << "copy back to the host" << std::endl;
    q_ct1.memcpy(&(results[0]), d_results, sizeof(Output) * results.size())
        .wait();
    std::cout << "vresults.size() = " << results.size() << std::endl;

    // free resources
    std::cout << "free the device memory" << std::endl;
    sycl::free(d_args, q_ct1);
    sycl::free(d_results, q_ct1);

    //test1();
    //std::vector<Output> results;

    return results;
}

/*
int main() {
    std::vector<DoFitArgs> v;
    auto tmp = doFitWrapper(v);
}
*/
