#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"
#include <atomic>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("CudaGraph capture & replay");
        {
            Matrix a = Matrix::ones(100, 100) * 2.0f;
            Matrix b = Matrix::ones(100, 100) * 3.0f;
            Matrix c(100, 100, MemoryMode::device_only);

            CudaGraph graph;
            ctx.check(!graph.is_capturing(), "initially not capturing");
            ctx.check(!graph.is_compiled(), "initially not compiled");

            // Warmup
            c.fill(0);
            add_(c, a);
            add_(c, b);
            c.synchronize();

            // Capture
            graph.begin_capture();
            ctx.check(graph.is_capturing(), "capturing state is true");
            c.fill(0);
            add_(c, a);
            add_(c, b);
            graph.end_capture();
            ctx.check(!graph.is_capturing(), "capturing ended");
            ctx.check(graph.is_compiled(), "graph is compiled");

            // Replay
            graph.replay();
            c.download();
            ctx.check_near(c.at(0, 0), 5.0f, 1e-5, "replayed graph computed correct result");
            ctx.check_near(c.at(99, 99), 5.0f, 1e-5, "replayed graph computed correct result at corner");
        }

        ctx.section("abandoned capture restores a usable stream");
        {
            Matrix out(2,2,MemoryMode::device_only);
            out.fill(0);synchronize();
            CudaGraph graph;graph.begin_capture();out.fill(3);graph.reset();
            ctx.check(!graph.is_capturing() && !graph.is_compiled(),"reset discards active capture");
            out.fill(7);out.download();
            ctx.check_near(out.at(0,0),7,0,"stream executes normally after abandoned capture");
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

        ctx.section("Pipeline isolates GEMM workspaces and restores stream after exceptions");
        {
            Matrix a=Matrix::ones(256,256),b=Matrix::ones(256,256);
            Matrix x(256,256,MemoryMode::device_only),y(256,256,MemoryMode::device_only);
            synchronize();Pipeline pipe(2);
            for(int i=0;i<3;++i) {
                pipe.enqueue([&]{multiply_into(a,b,x);});
                pipe.enqueue([&]{multiply_into(a,b,y);});
            }
            pipe.synchronize();x.download();y.download();
            ctx.check_near(x.at(150,200),256,0,"first stream GEMM correct");
            ctx.check_near(y.at(200,150),256,0,"second stream GEMM correct");
            auto old=compute_stream();
            ctx.check_throws<InvalidArgumentError>([&]{pipe.enqueue([]{throw InvalidArgumentError("test");});},"callback exception propagates");
            ctx.check(compute_stream()==old,"Pipeline restores stream on exception");
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
