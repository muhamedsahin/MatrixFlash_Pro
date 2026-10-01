#include "matrix_pro/matrix_pro.hpp"
#include <iostream>
using namespace matrix_pro;
int main() {
    try {
        Matrix a=Matrix::random(512,512,17),b=Matrix::random(512,512,23);
        Matrix bias=Matrix::ones(1,512),c(512,512,MemoryMode::device_only);
        GemmOptions options;
        options.precision=GemmPrecision::tf32; // explicit throughput/accuracy choice
        options.epilogue=GemmEpilogue::bias_relu;
        GemmPlan plan(512,512,512,options);
        auto tuning=plan.tune(a,b,c,1,0,&bias);
        plan.execute(a,b,c,1,0,&bias);synchronize();
        CudaGraph graph;
        graph.begin_capture();plan.execute(a,b,c,1,0,&bias);graph.end_capture();
        for(int i=0;i<10;++i)graph.replay();
        c.download();
        std::cout << "Tuned " << tuning.candidates_tested << " algorithms; output[0,0]=" << c.at(0,0) << '\n';
        Matrix coefficients(2,2,{4,1,1,3}),rhs(2,1,{1,2});
        LUFactorization factor(coefficients);
        Matrix solution=factor.solve(rhs);solution.download();
        std::cout << "Reused LU solution=" << solution.at(0,0) << ',' << solution.at(1,0)
                  << "; log|det|=" << factor.slogdet().log_abs_det << '\n';
    }catch(const std::exception& e){std::cerr << e.what() << '\n';return 1;}
}
