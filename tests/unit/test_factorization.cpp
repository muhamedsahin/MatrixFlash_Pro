#include "matrix_pro/matrix_pro.hpp"
#include "matrix_pro/ops/factorization.hpp"
#include "test_support.hpp"
#include <cmath>
#include <climits>
using namespace matrix_pro;
using namespace matrix_pro::test;
int main() {
    try {
        section("reused LU solves multiple RHS and transposed systems");
        Matrix a(3,3,{0,2,1,3,1,-1,1,0,4});
        Matrix b(3,2,{1,2,-3,4,5,-6});
        LUFactorization factor(a,2);
        for(bool trans:{false,true}) {
            Matrix x=factor.solve(b,trans);
            Matrix recovered=gemm(trans?a.transpose():a,x);
            recovered.download();
            for(size_t i=0;i<b.size();++i)check_near(recovered.data()[i],b.data()[i],2e-5,"LU solution residual");
            factor.solve_into(b,b,trans);b.download();
            recovered=gemm(trans?a.transpose():a,b);recovered.download();
            for(size_t i=0;i<recovered.size();++i)check(std::isfinite(recovered.data()[i]),"in-place RHS supported");
        }
        section("signed log determinant avoids FP32 overflow and preserves sign");
        Matrix d(2,2,{-1e20f,0,0,1e20f});auto result=slogdet(d);
        check(result.sign==-1,"negative determinant sign");
        check_near(result.log_abs_det,2*std::log(static_cast<double>(1e20f)),1e-6,"log determinant finite beyond float range");
        auto pivot=slogdet(Matrix(2,2,{0,2,3,0}));check(pivot.sign==-1,"pivot parity contributes determinant sign");
        check_near(pivot.log_abs_det,std::log(6.0),1e-6,"pivoted determinant magnitude");
        auto zero=slogdet(Matrix(0,0));check(zero.sign==1&&zero.log_abs_det==0,"empty determinant identity");
        Matrix singular(2,2,{1,2,2,4});LUFactorization bad(singular);
        check(bad.singular(),"singular detected");check(bad.slogdet().sign==0,"singular sign zero");
        check(std::isinf(bad.slogdet().log_abs_det)&&bad.slogdet().log_abs_det<0,"singular log -infinity");
        check_throws<SolverError>([&]{bad.solve(Matrix::ones(2,1));},"singular solve rejected");
        check_throws<ShapeMismatchError>([]{LUFactorization wrong(Matrix::ones(2,3));},"nonsquare coefficients rejected before factor allocation");
        Matrix capture_input=Matrix::identity(2);synchronize();
        CudaGraph capture;capture.begin_capture();
        check_throws<InvalidArgumentError>([&]{LUFactorization wrong(capture_input);},"LU construction rejected during capture");
        capture.reset();
        section("scaled matrix exponential and negative batched determinant");
        Matrix diagonal(2,2,{8,0,0,-8});Matrix ex=matrix_exp(diagonal);ex.download();
        check_near(ex.at(0,0),std::exp(8.0),std::exp(8.0)*2e-5,"exp large positive diagonal");
        check_near(ex.at(1,1),std::exp(-8.0),1e-8,"exp negative diagonal");
        Matrix rotation(2,2,{0,-12,12,0});ex=matrix_exp(rotation);ex.download();
        check_near(ex.at(0,0),std::cos(12.0),3e-6,"exp rotation cosine");
        check_near(ex.at(1,0),std::sin(12.0),3e-6,"exp rotation sine");
        Tensor batch({2,2,2},{0,2,3,0,2,0,0,3});auto dets=batch_det(batch);
        check_near(dets[0],-6,1e-5,"batch determinant retains negative sign");
        check_near(dets[1],6,1e-5,"batch positive determinant");
        Matrix identity=matrix_power(Matrix::identity(2),INT_MIN);identity.download();
        check_near(identity.at(0,0),1,0,"INT_MIN matrix exponent avoids signed overflow");
        return summary("FACTORIZATION");
    }catch(const std::exception& e){return fatal(e);}
}
