#include "gpu_function.h"

namespace GPU_FUNCTION_MAP{ 
    #if defined (__APPLE__)
    // std::unordered_map<KeyWithLibraryAndName, MTL::Function*, LibraryAndNameToFunction> compute_function_map;
    std::unordered_map<std::string, MTL::Function*> compute_function_map;
    // std::queue<MTL::Function*> compute_function_list;
    #endif
}


#ifdef __APPLE__




void gpuFunction::print_binding_info()
{
    auto arguments = this->get_function_arguments();
    fast_format("Function kernel binding info of {} \n    (index :          [type size] / [name])", func_name);
    for (int i = 0; i < arguments->count(); ++i) {
        MTL::Argument *arg = static_cast<MTL::Argument*>(arguments->object(i));
        if (arg->type() == MTL::ArgumentTypeBuffer) {
            fast_format("    Buffer index {:2} : {:2}   {} ", 
                arg->index(), arg->bufferDataSize(), ns_string_to_std_string(arg->name()) );
        }
    }
}

void gpuFunction::load(MTL::Library* source_library, std::string func_name){
    NS::Error* err;
    check_err(source_library, err);

    compute_function = source_library -> newFunction(std_string_to_ns_string(func_name));
    if(!compute_function){
        fast_print_err("function is empty", func_name);
    }     
    this->func_name = func_name;

    pipeline_reflection = nullptr;
    pipeline_descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
    pipeline_descriptor->setComputeFunction(compute_function);
    compute_pipeline = get_device()->newComputePipelineState(pipeline_descriptor, MTL::PipelineOptionArgumentInfo, &pipeline_reflection, &err);
    check_err(compute_pipeline, err);

    NS::Array* arguments = pipeline_reflection->arguments();
    need_to_bind_count = arguments->count();
    list_bind_buffer_size.resize(need_to_bind_count);

    for (int i = 0; i < need_to_bind_count; ++i) {
        MTL::Argument *arg = static_cast<MTL::Argument*>(arguments->object(i));
        if (arg->type() == MTL::ArgumentTypeBuffer) {
            list_bind_buffer_size[i] = arg->bufferDataSize();
        }
    }
}

void gpuFunction::load_with_multiple_entity(MTL::Library* source_library, std::string func_name, bool is_first){
    NS::Error* err;
    check_err(source_library, err);

    if(is_first){
        compute_function = source_library -> newFunction(std_string_to_ns_string(func_name));
        if(!compute_function){
            fast_print_err("function is empty");
        } 
        GPU_FUNCTION_MAP::compute_function_map.insert(std::make_pair(func_name, compute_function));
    }
    else{
        compute_function = GPU_FUNCTION_MAP::compute_function_map.at(func_name);
        if(!compute_function){
            fast_print_err("function is empty");
        } 
    }
    this->func_name = func_name;

    pipeline_reflection = nullptr;
    pipeline_descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
    pipeline_descriptor->setComputeFunction(compute_function);
    compute_pipeline = get_device()->newComputePipelineState(pipeline_descriptor, MTL::PipelineOptionArgumentInfo, &pipeline_reflection, &err);
    check_err(compute_pipeline, err);

    NS::Array* arguments = pipeline_reflection->arguments();
    need_to_bind_count = arguments->count();
    list_bind_buffer_size.resize(need_to_bind_count);

    for (int i = 0; i < need_to_bind_count; ++i) {
        MTL::Argument *arg = static_cast<MTL::Argument*>(arguments->object(i));
        if (arg->type() == MTL::ArgumentTypeBuffer) {
            list_bind_buffer_size[i] = arg->bufferDataSize();
        }
    }
    
}


#endif