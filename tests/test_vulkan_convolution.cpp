// SPDX-License-Identifier: MIT
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct Shape { int kernel, input_channels, output_channels, frames, dilation; };

static void check_convolution(ggml_backend_t backend, Shape shape) {
    const int K=shape.kernel, IC=shape.input_channels, OC=shape.output_channels;
    const int T=shape.frames, dilation=shape.dilation, pad=(K/2)*dilation;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({2*1024*1024, nullptr, true}), ggml_free);
    require(ctx != nullptr, "context allocation failed");
    auto weights=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,K,IC,OC);
    auto input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,T,IC);
    auto bias=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,OC);
    auto residual=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,T,OC);
    constexpr float in_scale=0.33333334f, in_slope=0.1f, out_slope=0.1f;
    auto output=ggml_conv_direct_1d_fused(ctx.get(),weights,input,bias,residual,
        pad,dilation,out_slope,in_scale,in_slope);
    require(ggml_backend_supports_op(backend,output), "Vulkan must support the tested F32 convolution");
    for(auto tensor : {weights,input,bias,residual}) ggml_set_input(tensor);
    ggml_set_output(output);
    auto graph=ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph,output);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),ggml_gallocr_free);
    require(alloc && ggml_gallocr_alloc_graph(alloc.get(),graph), "graph allocation failed");
    std::vector<float> w(K*IC*OC), x(T*IC), b(OC), r(T*OC), actual(T*OC);
    uint32_t state=0x12345678u;
    auto fill=[&](std::vector<float>& values){
        for(auto &value:values){state=1664525u*state+1013904223u;value=(float(state>>8)/16777216.0f-0.5f)*0.1f;}
    };
    fill(w);fill(x);fill(b);fill(r);
    ggml_backend_tensor_set(weights,w.data(),0,w.size()*sizeof(float));
    ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
    ggml_backend_tensor_set(bias,b.data(),0,b.size()*sizeof(float));
    ggml_backend_tensor_set(residual,r.data(),0,r.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
    ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
    double max_error=0;
    for(int oc=0;oc<OC;++oc) for(int t=0;t<T;++t){
        double sum=0;
        for(int ic=0;ic<IC;++ic) for(int k=0;k<K;++k){
            const int pos=t+k*dilation-pad;
            if(pos<0 || pos>=T) continue;
            float value=x[ic*T+pos]*in_scale;
            if(value<0) value*=in_slope;
            sum+=double(w[(oc*IC+ic)*K+k])*value;
        }
        sum+=b[oc];sum+=r[oc*T+t];if(sum<0)sum*=out_slope;
        require(std::isfinite(actual[oc*T+t]),"nonfinite result");
        max_error=std::max(max_error,std::abs(actual[oc*T+t]-sum));
    }
    require(max_error<2e-5,"convolution differs from the double-precision reference");
    std::printf("K=%d IC=%d OC=%d T=%d dilation=%d max_abs=%.3e\n",K,IC,OC,T,dilation,max_error);
}

int main(int argc, char **argv) {
    const char * name=argc>1 ? argv[1] : "Vulkan0";
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_init_by_name(name,nullptr),ggml_backend_free);
    if(!backend){std::fprintf(stderr,"Backend unavailable: %s\n",name);return 77;}
    try {
        // Partial time/channel tiles, all selected tile sizes, partial BK,
        // short clips with padding larger than T, and the largest model halo.
        const Shape cases[]={
            {3,3,15,1,1},{3,5,16,7,5},{7,9,17,33,3},{7,17,31,65,1},
            {11,7,32,129,5},{3,33,33,131,3},{11,32,64,129,5},
            {7,64,65,127,3},{3,128,128,17,1}
        };
        for(auto shape:cases) check_convolution(backend.get(),shape);
        std::printf("%s: 9 convolution cases passed\n",name);
    } catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
