#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>

#include "test1.h"

const int N = 6;
const int M = 2;

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

int test0() {
    show_platforms();

    queue q;
    buffer<int, 2> buf{range<2>{N, N}};

    q.submit([&](handler &h) {
        // prerequisites
        auto bufacc = buf.get_access<access::mode::read_write>(h);

        // kernel
        stream out{1024, 256, h};
        h.parallel_for(nd_range<3>{range<3>{N, N, 2}, range<3>{M, M, 2}},
            [=](nd_item<3> item){
                int i = item.get_global_id(0);
                int j = item.get_global_id(1);
                int k = item.get_global_id(2);
                int ii = item.get_local_id(0);
                int jj = item.get_local_id(1);
                int kk = item.get_local_id(2);
                out << k << "  " << kk << "\n";
                if (kk==0) 
                    bufacc[i][j] = ii+jj;
            });
    });

    auto bufacc1 = buf.get_access<access::mode::read>();
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++)
            std::cout << std::setw(10) << bufacc1[i][j] << "  ";
        std::cout << "\n";
    }

    return 0;
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
