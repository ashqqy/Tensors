# Scaled Dot-Product Attention

## Зависимости

- GTest
- [LibTorch](https://pytorch.org/get-started/locally/)

Torch необходим только для таргета correctness_tests.

## Сборка
```
cmake -S . -B ./build -DCMAKE_PREFIX_PATH="/путь/до/libtorch"
cmake --build ./build --config Release --target correctness_tests performance_tests -j $(nproc)
```
