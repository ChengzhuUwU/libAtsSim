#pragma once

#include "clock.h"
#include "shared_array.h"
// #include "parallel_task.h"
#include "utils.h"
#include "fence.h"
#include "metal_system.h"
#include "gpu_function.h"

const uint num_launcher = 15;
const uint num_fence = 1024;

constexpr uint get_num_fence() { return num_fence; };
sim::Fence& get_fence(uint idx = 0);
sim::Event& get_event(uint idx = 0);
sim::SharedEvent& get_shared_event(uint idx = 0);



class CommandList{

public:
    bool is_command_buffer_active = false;
    uint current_task_num = 0;
    enum PassType{
        PassTypeCompute,
        PassTypeBlit
    };
    PassType pass_type = PassTypeCompute;


public:
    #ifdef __APPLE__
    MTL::CommandQueue* command_queue;
    MTL::CommandBuffer* command_buffer; // command list

    MTL::ComputeCommandEncoder* compute_encoder;
    MTL::BlitCommandEncoder* blit_encoder;

    MTL::IndirectCommandBufferDescriptor* indirect_command_buffer_descriptor;
    MTL::IndirectCommandBuffer* indirect_command_buffer;
    
    #endif

public:
    inline void set_up_device(){
        #ifdef __APPLE__
        command_queue = get_device() -> newCommandQueue();
        if(!command_queue) std::cout << "Error : command queue is null\n";
        #endif
    }

    uint auto_fence_count = 0;

    inline void reset_auto_fence_count() { auto_fence_count = 0; }
    inline void make_fence_with_previous_cmd_buffer() {
        if (auto_fence_count != 0)  { wait_fence(get_fence(auto_fence_count)); }
        auto_fence_count++; if (auto_fence_count == get_num_fence() - 1)   {  auto_fence_count = 0;  }
        { make_fence(get_fence(auto_fence_count));  }
    }
    inline void send_last_cmd_buffer_in_list() {
        #ifdef __APPLE__
        is_command_buffer_active = false;
        current_task_num = 0;
        list_cmd_buffer.back()->commit();
        #endif
    }
    inline void segment_command_buffer(){
        if (auto_fence_count != 0) 
        { 
            wait_fence(get_fence(auto_fence_count)); 
        }
        if (auto_fence_count == get_num_fence() - 1) 
        { 
            auto_fence_count = 0;
        }
        { 
            auto_fence_count++;
            make_fence(get_fence(auto_fence_count));  
        } 
        send_list();
    }
    inline void segment_command_buffer_and_start_new(){
        segment_command_buffer();
        start_new_list();
    }

    #ifdef __APPLE__
    bool immidiate_send_mode;
    inline MTL::CommandBuffer* start_new_list(){
        if(!is_command_buffer_active){
            command_buffer = command_queue -> commandBuffer(); immidiate_send_mode = false;
            if(!command_buffer) std::cerr << "Error : command buffer is null\n";
            is_command_buffer_active = true;
        }
        else{
            std::cerr << "redundent call of new list\n";
        }

        return command_buffer;
    }
    
    std::vector<MTL::CommandBuffer*> list_cmd_buffer;
    inline MTL::CommandBuffer* start_new_list_with_new_buffer(){
        if (current_task_num != 0) {
            // make_fence_with_previous_cmd_buffer();
            fast_print_err("Previous Buffer Has Not Been Upload Yet");
            return command_buffer;
        }
        is_command_buffer_active = true;
        current_task_num = 0;
        command_buffer = command_queue -> commandBuffer(); immidiate_send_mode = true;
        list_cmd_buffer.push_back(command_buffer);
        return command_buffer;
    }
    inline void send_all_cmd_buffers() {
        for (auto& buffer : list_cmd_buffer) {
            buffer->commit();
        }
        current_task_num = 0;
        is_command_buffer_active = false;
    }
    inline void wait_all_cmd_buffers() {
        if (list_cmd_buffer.empty()) { fast_print_err("No Command Buffer In List"); return; }
        if (is_command_buffer_active || current_task_num) { fast_print_err("Exist Command Buffer That Does Not Upload Yet"); return; }
        list_cmd_buffer.back()->waitUntilCompleted();
        reset_auto_fence_count();
        clear_all_cmd_buffers();
    }
    inline void wait_all_cmd_buffers_but_not_clear_cmd_buffer() {
        if (list_cmd_buffer.empty()) { fast_print_err("No Command Buffer In List"); return; }
        if (is_command_buffer_active || current_task_num) { fast_print_err("Exist Command Buffer That Does Not Upload Yet"); return; }
        list_cmd_buffer.back()->waitUntilCompleted();
        reset_auto_fence_count();
    }
    inline std::vector<double> wait_all_cmd_buffers_and_get_costs(const bool get_kernel_time) {
        // for (auto& buffer : list_cmd_buffer) {
        //     buffer->waitUntilCompleted();
        // } 
        if (list_cmd_buffer.empty()) { fast_print_err("No Command Buffer In List"); return {}; }
        if (is_command_buffer_active || current_task_num) { fast_print_err("Exist Command Buffer That Does Not Upload Yet"); return {}; }
        list_cmd_buffer.back()->waitUntilCompleted();
        std::vector<double> list_costs(list_cmd_buffer.size());
        for (uint buffer_idx = 0; buffer_idx < list_cmd_buffer.size(); buffer_idx++)
        {
            auto& buffer = list_cmd_buffer[buffer_idx];
            list_costs[buffer_idx] = get_kernel_time ? get_command_buffer_cost_kernel(buffer_idx) : get_command_buffer_cost_gpu(buffer_idx);
        }
        reset_auto_fence_count();
        clear_all_cmd_buffers();
        return list_costs;
    }
    inline std::vector<double> get_cmd_buffer_costs(const bool get_kernel_time) {
        std::vector<double> list_costs(list_cmd_buffer.size());
        for (uint buffer_idx = 0; buffer_idx < list_cmd_buffer.size(); buffer_idx++)
        {
            auto& buffer = list_cmd_buffer[buffer_idx];
            list_costs[buffer_idx] = get_kernel_time ? get_command_buffer_cost_kernel(buffer_idx) : get_command_buffer_cost_gpu(buffer_idx);
        }
        return list_costs;
    }
    inline void clear_all_cmd_buffers(){
        list_cmd_buffer.clear();
    }

    inline MTL::CommandBuffer* get_command_buffer(){
        if(!command_buffer) std::cerr << "Error : command buffer is null\n";
        return command_buffer;
    }

    inline void print_status(MTL::CommandBuffer* input_buffer){
        auto status = input_buffer->status();
        switch (status) {
            case MTL::CommandBufferStatusNotEnqueued:
                fast_print("Not Enqueued");
                break;
            case MTL::CommandBufferStatusEnqueued:
                fast_print("Enqueued");
                break;
            case MTL::CommandBufferStatusCommitted:
                fast_print("Committed");
                break;
            case MTL::CommandBufferStatusScheduled:
                fast_print("Scheduled");
                break;
            case MTL::CommandBufferStatusCompleted:
                fast_print("Completed");
                break;
            case MTL::CommandBufferStatusError:
                fast_print("Error");
                break;
            default:
                fast_print("???");
                break;
            }
    }

    inline void print_status(){
        print_status(command_buffer);
    }

    inline double get_command_buffer_cost_kernel(){
        if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
            fast_print_err("Task Does Not Completed");
            return 0.0;
        }
        else {
            return (command_buffer->kernelEndTime() - command_buffer->kernelStartTime()) * 1000.0;
        }
    }
    inline double get_command_buffer_cost_gpu(){
        if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
            fast_print_err("Task Does Not Completed");
            return 0.0;
        }
        else {
            return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1000.0;
        }
    }
    inline double get_command_buffer_cost_kernel(uint buffer_idx){
        if (list_cmd_buffer[buffer_idx]->status() != MTL::CommandBufferStatusCompleted) {
            fast_print_err("Task Does Not Completed");
            return 0.0;
        }
        else {
            return (list_cmd_buffer[buffer_idx]->kernelEndTime() - list_cmd_buffer[buffer_idx]->kernelStartTime()) * 1000.0;
        }
    }
    inline double get_command_buffer_cost_gpu(uint buffer_idx){
        if (list_cmd_buffer[buffer_idx]->status() != MTL::CommandBufferStatusCompleted) {
            fast_print_err("Task Does Not Completed");
            return 0.0;
        }
        else {
            return (list_cmd_buffer[buffer_idx]->GPUEndTime() - list_cmd_buffer[buffer_idx]->GPUStartTime()) * 1000.0;
        }
    }

    inline MTL::IndirectCommandBuffer* start_new_list_indirect(){
        if(!is_command_buffer_active) start_new_list();

        indirect_command_buffer_descriptor -> setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatchThreads);
        indirect_command_buffer = get_device()->newIndirectCommandBuffer(indirect_command_buffer_descriptor, 1000000, MTL::ResourceOptionCPUCacheModeDefault);
        if(!indirect_command_buffer) std::cerr << "Error : indirect_command_buffer is null\n";
        return indirect_command_buffer;
    }

    template<MTL::DispatchType T = MTL::DispatchTypeSerial>
    inline MTL::ComputeCommandEncoder*& add_task(gpuFunction& function, uint pipeline_idx = 0){
        if(!is_command_buffer_active) start_new_list();
        pass_type = PassTypeCompute;
        current_task_num++;

        compute_encoder = command_buffer -> computeCommandEncoder(T);
        if(compute_encoder == nullptr) std::cerr << "Error : command encoder is null\n";
        // function.ref_encoder = std::make_shared<MTL::ComputeCommandEncoder>(command_encoder);
        function.ref_encoder = compute_encoder;
        function.begin();
        return compute_encoder;
    }


    inline void set_complete_hander(const MTL::HandlerFunction& func){
        // command_buffer -> addCompletedHandler( [&](MTL::CommandBuffer* buffer)->void{ fast_print("ok");} );
        command_buffer -> addCompletedHandler( func );
    }
    #else
    inline void add_task(gpuFunction& function, uint pipeline_idx = 0){
        pass_type = PassTypeCompute;
    }
    template<typename T>
    inline void add_reset_task_async(const T* input_array, uint fill_length){
        pass_type = PassTypeBlit;  
    }
    template<typename T>
    inline void add_reset_task_async(const SharedArray<T>& input_array){
        return add_reset_task_async((T*)input_array.buffer(), input_array.size());
    }
    inline uint get_command_buffer(){
        return 0;
    }
    inline uint start_new_list(){
        if(!is_command_buffer_active){
            is_command_buffer_active = true;
        }
        else{
            std::cerr << "redundent call of new list\n";
        }
        return 0;
    }
    #endif    
    

    inline bool send_list(){
        #ifdef __APPLE__
        if(is_command_buffer_active && current_task_num != 0){
            command_buffer -> commit();
            is_command_buffer_active = false;
            current_task_num = 0;
            return true;
        }
        else {
            return false;
        }
        #else
        return false;
        #endif
    }

    inline void wait(){
        #ifdef __APPLE__
        // if(current_task_num != 0)
        {
            command_buffer -> waitUntilCompleted(); 
            // current_task_num = 0;
        }
        #endif

        reset_auto_fence_count();

    }

    inline void send_and_wait(){
        // if (immidiate_send_mode)
        // {
        //     send_last_cmd_buffer_in_list();
        //     wait_all_cmd_buffers();
        // }
        // else 
        {
            bool send_success = send_list();
            if(send_success)
                wait();
        }
        
    }

    #ifdef __APPLE__

    inline MTL::CommandBufferStatus status(){
        // CommandBufferStatusNotEnqueued = 0,
        // CommandBufferStatusEnqueued = 1,
        // CommandBufferStatusCommitted = 2,
        // CommandBufferStatusScheduled = 3,
        // CommandBufferStatusCompleted = 4,
        // CommandBufferStatusError = 5,
        return command_buffer -> status();
    }

    template<typename T>
    inline MTL::BlitCommandEncoder* add_copy_task_async(
                                    SharedArray<T>& dst_array, uint output_offset, 
                                    const SharedArray<T>& src_array, uint input_offset, 
                                    uint copy_length){ 
        if(!is_command_buffer_active) start_new_list();
        pass_type = PassTypeBlit;
        current_task_num++;

        blit_encoder = command_buffer -> blitCommandEncoder();
        fast_check_null(blit_encoder, "blit_encoder");
        blit_encoder -> copyFromBuffer(src_array.buffer(), input_offset * sizeof(uint), 
                                       dst_array.buffer(), output_offset * sizeof(uint), 
                                       copy_length * sizeof(T));
        // blit_encoder -> synchronizeResource(input_array.buffer());
        blit_encoder -> endEncoding();
        return blit_encoder;
    }

    // template<MTL::DispatchType T = MTL::DispatchTypeSerial>
    // inline MTL::ComputeCommandEncoder*& add_task(gpuFunction& function, uint pipeline_idx = 0){
    //     if(!is_command_buffer_active) start_new_list();
    //     pass_type = PassTypeCompute;
    //     current_task_num++;
    
    //     compute_encoder = command_buffer -> computeCommandEncoder(T);
    //     if(compute_encoder == nullptr) std::cerr << "Error : command encoder is null\n";
    //     function.ref_encoder = compute_encoder;
    //     function.begin();
    //     return compute_encoder;
    // }

    template<typename T>
    inline MTL::BlitCommandEncoder* add_copy_task_async(SharedArray<T>& dst_array, const SharedArray<T>& src_array){ 
        return add_copy_task_async(dst_array, 0, src_array, 0, src_array.size());
    }

    template<typename T>
    inline MTL::BlitCommandEncoder* add_aync_task(const SharedArray<T>& input_array){ 
        if(!is_command_buffer_active) start_new_list();
        pass_type = PassTypeBlit;
        current_task_num++;

        blit_encoder = command_buffer -> blitCommandEncoder();
        fast_check_null(blit_encoder, "blit_encoder");
        blit_encoder -> synchronizeResource(input_array.buffer());
        blit_encoder -> endEncoding();
        return blit_encoder;
    }

    #endif


    #ifdef __APPLE__
private:
    
    template<typename T>
    inline void fn_fill_buffer(const SharedArray<T>& input_array){ 
        blit_encoder -> fillBuffer(input_array.buffer(), NS::Range(0, input_array.get_byte_size()), 0);
    }
    template<typename T>
    inline void fn_fill_buffer(const T* input_array, uint fill_length){ 
        static_assert(!std::is_same<T, MTL::Buffer>::value, "Type Wrong : You are trying to fill buffer with size of length * sizeof(MTL::Buffer)");
        blit_encoder -> fillBuffer((MTL::Buffer*)(input_array), NS::Range(0, fill_length * sizeof(T)), 0);
    }

public:
    template<typename T>
    inline MTL::BlitCommandEncoder* add_reset_task_async(const T* input_array, uint fill_length){
        if(!is_command_buffer_active) start_new_list();
        pass_type = PassTypeBlit;
        current_task_num++;
        
        blit_encoder = command_buffer -> blitCommandEncoder();
        fast_check_null(blit_encoder, "blit_encoder");
        fn_fill_buffer(input_array, fill_length);
        blit_encoder -> endEncoding();
        return blit_encoder;
    }

    // template<MTL::DispatchType T = MTL::DispatchTypeSerial>
    // inline MTL::ComputeCommandEncoder*& add_task(gpuFunction& function, uint pipeline_idx = 0){
    //     if(!is_command_buffer_active) start_new_list();
    //     pass_type = PassTypeCompute;
    //     current_task_num++;
    //     compute_encoder = command_buffer -> computeCommandEncoder(T);
    //     if(compute_encoder == nullptr) std::cerr << "Error : command encoder is null\n";
    //     function.ref_encoder = compute_encoder;
    //     function.begin(pipeline_idx);
    //     return compute_encoder;
    // }

    template<typename T>
    inline MTL::BlitCommandEncoder* add_reset_task_async(const SharedArray<T>& input_array){
        return add_reset_task_async((T*)input_array.buffer(), input_array.size());
    }
    #endif

public:

    // 使用 sim::Fence& 有误！
    inline void wait_fence(sim::Fence fence){
        #if defined (__APPLE__)
        if(pass_type == PassTypeCompute)   {  compute_encoder -> waitForFence(fence.fence);  }
        else if(pass_type == PassTypeBlit) {     blit_encoder -> waitForFence(fence.fence);  }
        #endif
    }
    inline void make_fence(sim::Fence fence){
        #if defined (__APPLE__)
        if(pass_type == PassTypeCompute)   {  compute_encoder -> updateFence(fence.fence); }
        else if(pass_type == PassTypeBlit) {     blit_encoder -> updateFence(fence.fence); }
        #endif
    }
    inline void wait_event(sim::Event event, uint64_t value){
        #if defined (__APPLE__)
        command_buffer->encodeWait(event.event, value);
        #endif
    }
    inline void make_event(sim::Event event, uint64_t value){
        #if defined (__APPLE__)
        command_buffer->encodeSignalEvent(event.event, value);
        #endif
    }
    inline void wait_cpu(sim::SharedEvent event, uint64_t value){
        #if defined (__APPLE__)
        if (value == 0) fast_format_err("Waiting event should larger than 1");
        // if (value > 61) fast_format_err("Waiting event should smaller than 60 due to hardware limitation");
        command_buffer->encodeWait(event.event, value);
        #endif
    }

// public:
//     uint auto_fence_count = 0;
//     inline void refresh_fence_count() { auto_fence_count = 0; }
//     inline void begin_cluster() {
//         start_new_list();
//     }
//     inline void end_cluster() {
//         if(auto_fence_count != 0)  { wait_fence(get_fence(auto_fence_count - 1)); }
//         { make_fence(get_fence(auto_fence_count));  } 
//     }

public:

    #if defined (__APPLE__)
    template<typename Func>
    double loop_for(Func function, uint loop_count = 100){
        // SimClock clock; clock.start_clock();
        auto buffer = this->start_new_list();
        for (int i = 0; i < 100; i++) { function(); }
        this->send_and_wait();
        auto gpu_time = (buffer->GPUEndTime() - buffer->GPUStartTime()) * 1000;
        return gpu_time;
        // return clock.end_clock(false);
    }
    #else
    template<typename Func>
    double loop_for(Func function, uint loop_count = 100){
        return 0;
    }
    #endif
    
};



void init_command_list();
CommandList& get_command_list(uint idx = 0);
gpuFunction& add_compute_task(gpuFunction& func, uint queue_idx = 0);
