#include "matrix_pro/matrix_pro.hpp"
#include "matrix_pro/ops/gemm.hpp"
#include "test_support.hpp"
#include <climits>
#include <cmath>
#include <future>
#include <limits>
#include <vector>
using namespace matrix_pro;
using namespace matrix_pro::test;

int main() {
    try {
        section("transpose, scaling, accumulation, and fused epilogues");
        for (bool ta:{false,true}) for(bool tb:{false,true}) for(int ep=0;ep<4;++ep) {
            const int m=7,n=9,k=5;
            std::vector<float> av(m*k),bv(k*n),biasv(n);
            for(size_t i=0;i<av.size();++i)av[i]=(static_cast<int>(i%11)-5)*0.17f;
            for(size_t i=0;i<bv.size();++i)bv[i]=(static_cast<int>(i%7)-3)*0.23f;
            for(int i=0;i<n;++i)biasv[i]=(i-4)*0.11f;
            Matrix a(ta?k:m,ta?m:k,av),b(tb?n:k,tb?k:n,bv),bias(1,n,biasv);
            Matrix c=Matrix::ones(m,n);
            GemmOptions o; o.transpose_left=ta;o.transpose_right=tb;o.epilogue=static_cast<GemmEpilogue>(ep);
            GemmPlan p(m,n,k,o);p.execute(a,b,c,0.7f,-0.3f,ep?&bias:nullptr);
            check(!c.host_current(),"execute invalidates host mirror");c.download();
            for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
                double dot=0;
                for(int t=0;t<k;++t)dot+=static_cast<double>(av[ta?t*m+i:i*k+t])*bv[tb?j*k+t:t*n+j];
                double ref=0.7f*dot-0.3f;
                if(ep)ref+=biasv[j];
                if(ep==2)ref=std::max(0.0,ref);
                if(ep==3)ref=0.5*ref*(1+std::tanh(0.7978845608*(ref+0.044715*ref*ref*ref)));
                check_near(c.at(i,j),ref,3e-5,"GEMM agrees with independent double dot product");
            }
        }
        section("empty inner dimension, alias rejection and validation");
        Matrix a(3,0),b(0,4),c=Matrix::ones(3,4),bias=Matrix::ones(1,4);
        GemmOptions o;o.epilogue=GemmEpilogue::bias_relu;
        GemmPlan empty(3,4,0,o);empty.execute(a,b,c,1,-2,&bias);c.download();
        for(float x:c.data())check_near(x,0,0,"empty K applies beta and epilogue");
        empty.execute(a,b,c,1,0,&bias);c.download();for(float x:c.data())check_near(x,1,0,"empty K zero beta");
        Matrix square=Matrix::identity(3);
        GemmPlan p(3,3,3);
        check_throws<InvalidArgumentError>([&]{p.execute(square,square,square);},"reject output alias");
        check_throws<ShapeMismatchError>([&]{p.execute(square,b,c);},"reject operand mismatch");
        check_throws<InvalidArgumentError>([]{GemmPlan huge(static_cast<size_t>(INT_MAX)+1,1,1);},"reject int overflow before allocation");
        check_throws<InvalidArgumentError>([]{allocate_device_memory(std::numeric_limits<size_t>::max());},"reject memory-pool size-class overflow without looping");
        GemmPlan moved=std::move(p);
        check_throws<InvalidArgumentError>([&]{p.execute(square,square,square);},"reject moved-from plan");

        section("tuning preserves beta input and captures without allocation");
        Matrix left=Matrix::random(128,128,17),right=Matrix::random(128,128,23);
        Matrix out=Matrix::ones(128,128);
        GemmOptions fast;fast.precision=GemmPrecision::tf32;
        GemmPlan plan(128,128,128,fast);
        auto tuning=plan.tune(left,right,out,1,0.5f);
        check(tuning.candidates_tested>0,"tuning measured candidates");
        out.download();for(float x:out.data())check_near(x,1,0,"tuning leaves output unchanged");
        Matrix reference=gemm(left,right);
        reference.download();
        plan.execute(left,right,out);
        synchronize();
        CudaGraph graph;graph.begin_capture();plan.execute(left,right,out);graph.end_capture();
        graph.replay();out.download();
        for(size_t i=0;i<out.size();++i)check_near(out.data()[i],reference.data()[i],0.025,"TF32 graph agrees with FP32 reference");
        right.fill(0);graph.replay();out.mark_host_stale();out.download();
        for(float x:out.data())check_near(x,0,0,"graph reads updated input buffers");
        section("vector kernels and long vector dispatch preserve full FP32 input precision");
        for(size_t inner:{size_t(127),size_t(2048)}) {
        Matrix vector=Matrix::random(1,inner,91),weights=Matrix::random(inner,127,97);
        for(size_t i=0;i<vector.size();++i)vector.data()[i]-=0.5f;
        for(size_t i=0;i<weights.size();++i)weights.data()[i]-=0.5f;
        vector.upload();weights.upload();
        Matrix product=multiply(vector,weights);product.download();
        for(size_t j=0;j<weights.cols();++j) {
            double reference=0;
            for(size_t k=0;k<weights.rows();++k)reference+=static_cast<double>(vector.data()[k])*weights.data()[k*weights.cols()+j];
            check_near(product.at(0,j),reference,2e-5,"long vector agrees with strict double-dot tolerance");
        }
        }
        section("default cached GEMM across concurrent threads");
        auto worker=[](float value){
            Matrix x=Matrix::ones(256,256)*value,y=Matrix::ones(256,256),z(256,256,MemoryMode::device_only);
            for(int i=0;i<10;++i)multiply_into(x,y,z);
            z.download();return z.at(127,129);
        };
        auto f=std::async(std::launch::async,worker,2.0f),g=std::async(std::launch::async,worker,-3.0f);
        check_near(f.get(),512,0.01,"thread one workspace isolated");check_near(g.get(),-768,0.01,"thread two workspace isolated");
        return summary("GEMM_PLAN");
    } catch(const std::exception& e){return fatal(e);}
}
