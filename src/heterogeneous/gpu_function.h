#pragma once

#include "atomic.h"
#include "float_n.h"
#include "make_arguments.h"
#include "shared_array.h"
#include "utils.h"

#if __APPLE__
inline NS::String* std_string_to_ns_string(std::string str){
    return NS::String::string(str.c_str(), NS::UTF8StringEncoding);
}

inline NS::String* char_to_ns_string(const char* str){
    return NS::String::string(str, NS::UTF8StringEncoding);
}

inline const std::string ns_string_to_std_string(const NS::String* str){
    return std::string(str->utf8String());
}

namespace GPU_FUNCTION_MAP{
    struct KeyWithLibraryAndName {
        MTL::Library* key1;
        std::string key2;
        bool operator==(const KeyWithLibraryAndName& other) const {
            return key1 == other.key1 && key2 == other.key2;
        }
    };
    struct LibraryAndNameToFunction {
        std::size_t operator()(const KeyWithLibraryAndName& k) const {
            return std::hash<MTL::Library*>()(k.key1) ^ std::hash<std::string>()(k.key2);
        }
    };

    // extern std::unordered_map<KeyWithLibraryAndName, MTL::Function*, LibraryAndNameToFunction> compute_function_map;
}

#endif

// 获取成员在类中的偏移量
template<typename StructType, typename MemberType>
inline uint get_offset_in_struct(const StructType& object, const MemberType& member){
    return (uint)(
        (char*)(&member) - 
        (char*)(&object));
}


class gpuFunction{
    using size_t = uint;
    
private:
    #define NUM_PIPELINE 1
    #define FOR_PIPELINE for(uint i = 0; i < NUM_PIPELINE; i++)

public:
    #ifdef __APPLE__
    // std::shared_ptr<MTL::ComputeCommandEncoder> ref_encoder;
    // std::shared_ptr<CommandList>  ref_command_list;
    MTL::ComputeCommandEncoder* ref_encoder;
    MTL::Function* compute_function;
    MTL::ComputePipelineState* compute_pipeline; // [NUM_PIPELINE]
    MTL::ComputePipelineReflection *pipeline_reflection;
    MTL::ComputePipelineDescriptor *pipeline_descriptor;
    #endif

    std::string func_name;

private:
    size_t need_to_bind_count = 0;
    size_t bind_buffer_count = 0;
    size_t bind_constant_count = 0;

    std::vector<uint> list_bind_buffer_size;

private:
    #ifdef __APPLE__
    MTL::Size total_threads = MTL::Size::Make(1, 1, 1);
    MTL::Size block_dim     = MTL::Size::Make(256, 1, 1);
    MTL::Size grid_dim      = MTL::Size::Make(1, 1, 1);
    #endif

public:
    #if defined (__APPLE__)
    void load(MTL::Library* source_library, std::string func_name);
    void load_with_multiple_entity(MTL::Library* source_library, std::string func_name, bool is_first);
    #endif

    gpuFunction() {};
    gpuFunction(const gpuFunction& func){
        #if defined (__APPLE__)
        // FOR_PIPELINE{
        //     compute_pipeline[i] = func.compute_pipeline[i];
        // }
        compute_pipeline = func.compute_pipeline;
        ref_encoder = func.ref_encoder;
        #endif
    }

private:
    #ifdef __APPLE__
    bool use_argument_buffer = false;
    std::vector<MTL::Buffer*> argument_buffer_list;
    std::vector<uint> argument_buffer_offset_list;
    std::vector<MTL::Buffer*> binding_list;
    std::vector<MTL::ResourceUsage> visibal_type_list;
    #endif

public:
    #ifdef __APPLE__

    // template<typename T, typename... Args>
    // void set_buffer_visibal(SharedArray<T>& arr, Args&... array_list){
    //     set_buffer_visibal(arr);
    //     set_buffer_visibal(array_list...);
    // }

    const std::string get_buffer_name_in_offset(const uint index){
        
        if (index < need_to_bind_count) {
            MTL::Argument *arg = static_cast<MTL::Argument*>(pipeline_reflection->arguments()->object(index));
            return ns_string_to_std_string(arg->name());
        }
        else {
            return std::string("");
        }
        
    }
    uint get_buffer_type_size_in_offset(const uint index){
        return list_bind_buffer_size[index];
    }

    NS::Array* get_function_arguments()
    {
        return pipeline_reflection->arguments();
    }
    void print_binding_info();
    
    template<typename T>
    void set_buffer_visibal(T* buffer, AccessType access_type = AccessTypeReadWrite){
        // if((MTL::Buffer*)(buffer) == nullptr){
        //     std::cerr << " ------- " << func_name << " : Buffer is empty , at offset << " << binding_list.size() << " ------- \n";
        //     return;
        // }

        binding_list.push_back((MTL::Buffer*)(buffer));
        if(access_type == AccessTypeRead) visibal_type_list.push_back(MTL::ResourceUsageRead);
        else if(access_type == AccessTypeWrite) visibal_type_list.push_back(MTL::ResourceUsageWrite);
        else if(access_type == AccessTypeReadWrite) 
        visibal_type_list.push_back(MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        else std::cerr << "binding type error\n";
        
    }

    template<typename T>
    inline void set_buffer_visibal(SharedArray<T>& arr, AccessType access_type = AccessTypeReadWrite){
        // if(arr.buffer() == nullptr){
        //     std::cerr << " ------- " << func_name << " : Buffer is empty , at offset << " << binding_list.size() << " ------- \n";
        // }
        set_buffer_visibal(arr.buffer(), access_type);
    }

    // get from structure
    void set_buffer_visibal(){
        // 
    }

    inline void add_argument_buffer(MTL::Buffer* buffer, uint offset = 0){
        argument_buffer_list.push_back(buffer);
        argument_buffer_offset_list.push_back(offset);
        use_argument_buffer = true;
    }

    template<typename T>
    inline void add_argument_buffer(SharedArray<T>& arr, uint inner_buffer_offset = 0){
        add_argument_buffer(arr.buffer(), inner_buffer_offset);
    }

    // template<typename T, typename... Args>
    // void add_argument_buffer(SharedArray<T>& arr, Args&... array_list){
    //     add_argument_buffer(arr);
    //     add_argument_buffer(array_list...);
    // }
    #else

    template<typename T>
    inline void set_buffer_visibal(SharedArray<T>& arr){
    }
    template<typename T>
    void set_buffer_visibal(T* buffer, AccessType access_type){
    }
    template<typename T>
    inline void add_argument_buffer(SharedArray<T>& arr, uint offset = 0){
    }

    #endif
//     template<typename ArgumentStructure, typename BindFunc>
//     void init_arguments(ArgumentStructure& cpu_struct, BindFunc bind_function){

//         uint buffer_length = sizeof(ArgumentStructure);
//         argument_buffer = get_device()->newBuffer(buffer_length, MTL::ResourceStorageModeShared);
//         ArgumentStructure& gpu_struct = static_cast<ArgumentStructure*>(argument_buffer -> contents())[0];
        
//         bind_function(gpu_struct, cpu_struct);

//         use_argument_buffer = true;

//         // // Metal 2.x
//         // MTL::ArgumentEncoder* encoder = compute_function -> newArgumentEncoder(0);
//         // uint buffer_length = encoder -> encodedLength();
//         // argument_buffer = get_device() -> newBuffer(buffer_length, MTL::ResourceStorageModeShared);
//         // // argument encoder binding
//         // encoder -> setArgumentBuffer(argument_buffer, 0);
//         // bind_func(encoder); 
//     }

    #ifdef __APPLE__
    void bind_self(){
        for (uint i = 0; i < argument_buffer_list.size(); i++) {
            auto& argument_buffer = argument_buffer_list[i];
            if(argument_buffer -> allocatedSize() == 0) std::cerr << "buffer size is zero !!!";
            ref_encoder -> setBuffer(argument_buffer, argument_buffer_offset_list[i], bind_buffer_count++);
        }
        if(visibal_type_list.empty()){
            for(uint i = 0; i < binding_list.size(); i++){
                const auto& buffer = binding_list[i]; // MTL::Buffer*
                if(buffer == nullptr){
                    std::cerr << " ------- " << func_name << " : Buffer is empty , at offset << " << binding_list.size() << " ------- \n";
                    return;
                }
                ref_encoder -> useResource(binding_list[i], MTL::ResourceUsageWrite | MTL::ResourceUsageRead);
            }
        }
        else{
            for(uint i = 0; i < binding_list.size(); i++){
                ref_encoder -> useResource(binding_list[i], visibal_type_list[i]);
                // ref_encoder -> useResource(binding_list[i], MTL::ResourceUsageWrite | MTL::ResourceUsageRead);
            }
        }
        
        // if(argument_buffer->allocatedSize() == 0) std::cerr << "buffer size is zero !!!";
        // ref_encoder -> setBuffer(argument_buffer, 0, bind_buffer_count++);
    }
    #endif

//     // 不可以用于读取数据（因为存的是gpu的指针）
//     MTL::Buffer* get_argument_buffer(){
//         return argument_buffer;
//     }



public:

    #ifdef __APPLE__
    inline void set_total_threads(size_t input_total_threads){
        total_threads.width = (NS::UInteger)input_total_threads;
        set_grid_dim((total_threads.width + block_dim.width - 1) / block_dim.width);
    }
    inline void set_block_dim(size_t input_block_dim) { block_dim.width = (NS::UInteger)input_block_dim; }
    inline void set_grid_dim(size_t input_grid_dim)   { grid_dim.width = (NS::UInteger)input_grid_dim; }
    inline void set_block_dim(size_t dim_x, size_t dim_y)  { block_dim.width = (NS::UInteger)dim_x; block_dim.height = (NS::UInteger)dim_y;}
    inline void set_grid_dim(size_t dim_x, size_t dim_y)   { grid_dim.width = (NS::UInteger)dim_x; grid_dim.height = (NS::UInteger)dim_y;}
    inline uint get_grid_dim()   { return grid_dim.width; }
    #endif
    
public:
    #ifdef __APPLE__
    // Dispatch
    inline void begin(){
        bind_buffer_count = 0;
        // ref_encoder -> setComputePipelineState(compute_pipeline[idx]);
        ref_encoder -> setComputePipelineState(compute_pipeline);
        if(use_argument_buffer){
            bind_self();
        }
    }

    inline MTL::ArgumentEncoder* begin_indirect(){
        bind_buffer_count = 0;
        return compute_function -> newArgumentEncoder(0);
    }

    template<typename T> 
    inline void bind_ptr(const SharedArray<T>& array, size_t prefix = 0){
        check_binding_type<T>();
        if(array.size() == 0) fast_format_err("Buffer Is Empty In Kernel {} : {}", this->func_name, bind_buffer_count);
        
        ref_encoder -> setBuffer(array.buffer(), prefix * sizeof(T), bind_buffer_count++);
    }
    template<typename T> 
    inline void bind_ptr(const T* ptr, size_t prefix = 0){
        check_binding_type<T>();
        if(ptr == nullptr) fast_format_err("Buffer Is Empty In Kernel {} : {}", this->func_name, bind_buffer_count);
        ref_encoder -> setBuffer((MTL::Buffer*)ptr, prefix * sizeof(T), bind_buffer_count++);
    }

    // template<typename... Args> 
    // inline void bind_ptrs(Args... args){  bind_ptr(args...);  }

    // template<typename T> 
    // inline void bind_ptr(const SharedArray<T>& array, size_t prefix){
    //     if(array.size() == 0) std::cerr << "buffer size is zero !!!";
    //     ref_encoder -> setBuffer(array.buffer(), prefix * sizeof(T), bind_buffer_count++);
    // }
    // template<typename T> 
    // inline void bind(const AtomicArray<T>& array, size_t prefix){
    //     if(array.size() == 0) std::cerr << "buffer size is zero !!!";
    //     ref_encoder -> setBuffer(array.buffer(), prefix * sizeof(T), bind_buffer_count++);
    // }

    template <typename T>
    inline bool check_binding_type()
    {
        if (sizeof(T) != get_buffer_type_size_in_offset(bind_buffer_count) 
            && !std::is_same<T, FlagType>()) // atomic_bool in gpu is 4 bytes
        {
            fast_print_err(std::format("({}) Type does not correspond at offset {} : Expect for {} , Actually for {} ({})", 
                func_name, bind_buffer_count, get_buffer_type_size_in_offset(bind_buffer_count), sizeof(T), get_buffer_name_in_offset(bind_buffer_count)));
            return false;
        }
        else {
            return true;
        }
    }
    inline bool check_binding_count()
    {
        if (bind_buffer_count != need_to_bind_count) {
            fast_print_err(
                std::format("({}) Buffer binding count does not correspond with buffer : Expect for {} , Actually for {} . Buffers is ",
                    func_name, need_to_bind_count, bind_buffer_count));
            
            auto arguments = get_function_arguments();
            for (int i = 0; i < arguments->count(); ++i) {
                MTL::Argument *arg = static_cast<MTL::Argument*>(arguments->object(i));
                if (arg->type() == MTL::ArgumentTypeBuffer) {
                    std::cout << "    Buffer index : " << arg->index() << " : " << ns_string_to_std_string(arg->name()) << std::endl;
                }
            }
            return false;
        }
        else {
            return true;
        }
    }

    template<typename T>
    inline void bind_constant(const T& constant_value){
        check_binding_type<T>();
        ref_encoder -> setBytes(&constant_value, sizeof(constant_value), bind_buffer_count++);
    }

    template<typename T>
    inline void bind_texture(MTL::Texture* texture, size_t prefix){
        ref_encoder -> setTexture(texture, prefix);
    }

    // 默认dispatch线程
    inline void launch_async(size_t input_total_threads){
        check_binding_count();
        if(input_total_threads != total_threads.width){
            set_total_threads(input_total_threads);
        }
        if(input_total_threads > 2000000) {
            fast_print("too many thread!!");
            return;
        }
        ref_encoder -> dispatchThreads(total_threads, block_dim);
        ref_encoder -> endEncoding();
        bind_buffer_count = 0;
    }
    // 给定线程数和blockDim
    inline void launch_async(size_t input_total_threads, size_t input_block_dim){

        check_binding_count();

        if(input_total_threads != total_threads.width){
            set_total_threads(input_total_threads);
            set_block_dim(input_block_dim);
            set_grid_dim(input_total_threads / input_block_dim);
        }

        // ref_encoder -> dispatchThreadgroups(grid_dim, block_dim);
        ref_encoder -> dispatchThreads(total_threads, block_dim);
        // fast_print(total_threads.width);
        ref_encoder -> endEncoding();
        bind_buffer_count = 0;
    }

    // From Argument Buffer
    inline void launch_async(const Int4* indirect_command_buffer, uint idb_offset = 0, uint input_dim = 256){
        // if(indirect_command_buffer[0].x > 10000) {
        //     fast_print("too many thread!!");
        //     return;
        // }
        check_binding_count();
        set_block_dim(input_dim);
        ref_encoder -> dispatchThreadgroups((MTL::Buffer*)indirect_command_buffer, idb_offset * sizeof(Int4), block_dim);
        ref_encoder -> endEncoding();
        bind_buffer_count = 0;
    }
    // From SharedArray
    inline void launch_async(SharedArray<Int4>& indirect_command_buffer, uint idb_offset, uint input_dim = 256){
        if(indirect_command_buffer.size() == 0){
            fast_print("indirect buffer if null !");
            return;
        }
        launch_async((Int4*)indirect_command_buffer.buffer(), idb_offset, input_dim);
    }


    

    #define NUM_SM 30
    #define WARP_DIM 32
    #define WARP_NUM_PER_SM 48
    #define BLOCK_NUM_PER_SM 8
    #define BLOCK_NUM NUM_SM * BLOCK_NUM_PER_SM
    #define WARP_NUM_PER_BLOCK WARP_NUM_PER_SM / BLOCK_NUM_PER_SM

    inline void launch_persistent(size_t input_block_dim = 256){
        
        if(grid_dim.width != NUM_SM){
            set_block_dim(WARP_DIM, WARP_NUM_PER_BLOCK);
            set_grid_dim(NUM_SM);
        }
        
        ref_encoder->dispatchThreadgroups(grid_dim, block_dim);
        ref_encoder -> endEncoding();
        bind_buffer_count = 0;
    }
    #else
    inline void begin(uint idx = 0){
        bind_buffer_count = 0;
    }
    template<typename T> 
    inline void bind_ptr(const SharedArray<T>& array, size_t prefix = 0){
        if(array.size() == 0) std::cerr << "buffer size is zero !!!";
    }
    template<typename T> 
    inline void bind_ptr(const T* ptr, size_t prefix = 0){
        if(ptr == nullptr) std::cerr << "buffer size is zero !!!";
    }
    template<typename T>
    inline void bind_constant(const T& constant_value){
    }
    inline void launch_async(size_t input_total_threads){
        bind_buffer_count = 0;
    }
    inline void launch_async(size_t input_total_threads, size_t input_block_dim){
        bind_buffer_count = 0;
    }
    // uint idb_offset = 0, uint input_dim = 256
    inline void launch_async(const Int4* indirect_command_buffer, uint idb_offset = 0, uint input_dim = 256){
    }
    // From SharedArray
    inline void launch_async(SharedArray<Int4>& indirect_command_buffer, uint idb_offset, uint input_dim = 256){
        if(indirect_command_buffer.size() == 0){
            fast_print("indirect buffer if null !");
            return;
        }
    }
    #endif

public:
    inline void generate_indirect_buffer(){
        // MTL::IndirectCommandBufferDescriptor* icbDescriptor;
        // icbDescriptor->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatchThreads);
        // icbDescriptor->setMaxKernelBufferBindCount(10);
        // MTL::IndirectCommandBuffer* indirect_command_buffer = get_device()->newIndirectCommandBuffer(,1, MTL::ResourceStorageModeShared)

    }

};
