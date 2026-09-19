#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("CudaGraph capture & replay");
        {
            Matrix a(100, 100, 2.0f);
            Matrix b(100, 100, 3.0f);
            Matrix c(100, 100);

            CudaGraph graph;
            ctx.check(!graph.is_capturing(), "initially not capturing");
            ctx.check(!graph.is_compiled(), "initially not compiled");

            // Warmup
            c = a + b;
            c.synchronize();

            // Capture
            graph.begin_capture();
            ctx.check(graph.is_capturing(), "capturing state is true");
            c = a + b;
            graph.end_capture();
            ctx.check(!graph.is_capturing(), "capturing ended");
            ctx.check(graph.is_compiled(), "graph is compiled");

            // Replay
            graph.replay();
            c.download();
            ctx.check_near(c.at(0, 0), 5.0f, 1e-5, "replayed graph computed correct result");
            ctx.check_near(c.at(99, 99), 5.0f, 1e-5, "replayed graph computed correct result at corner");
        }

        ctx.section("Pipeline multi-stage execution");
        {
            Pipeline pipe(4);
            ctx.check(pipe.num_stages() == 4, "4 stages in pipeline");

            std::atomic<int> counter{0};
            for (int i = 0; i < 8; ++i) {
                pipe.enqueue([&counter]() {
                    counter.fetch_add(1);
                });
            }
            pipe.synchronize();
            ctx.check(counter.load() == 8, "all 8 tasks completed across pipeline stages");
        }

        ctx.section("ExecutionProfiler timing");
        {
            ExecutionProfiler profiler;
            profiler.begin("matmul_timing");
            Matrix a = Matrix::ones(256, 256);
            Matrix b = Matrix::ones(256, 256);
            Matrix c = a * b;
            c.synchronize();
            profiler.end();

            ctx.check(profiler.last_elapsed_ms() > 0.0f, "profiler recorded non-zero elapsed time");
            ctx.check(profiler.total_ms() > 0.0f, "total_ms is non-zero");
            ctx.check(!profiler.results().empty(), "results list is not empty");
            ctx.check(!profiler.summary().empty(), "summary string is generated");
        }

        return ctx.summary("EXECUTION");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

