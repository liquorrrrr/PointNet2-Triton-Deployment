#include <torch/extension.h>
#include "ball_query.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query", &ball_query, "Ball Query CUDA implementation");
    m.def("furthest_point_sampling", &furthest_point_sampling, "FPS CUDA implementation");
}