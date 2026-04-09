#include "tensor.hpp"

int main() {
    std::vector<float> v1 = {2, 2, 2, 3, 3, 3};
    std::vector<float> v2 = {4, 4, 4, 2, 2, 2};
    Tensor t1(1, 2, 3, v1.begin(), v2.end());
    Tensor t2(1, 3, 2, v2.begin(), v2.end());
    Tensor t3(1, 2, 2);
    Tensor::multiply_2d(t1, t2, t3, 0);

    std::cout << t1 << "*" << std::endl;
    std::cout << t2 << "=" << std::endl;
    std::cout << t3 << std::endl;
}
