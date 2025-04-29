#include <iostream>
#include <tbb/tbb.h>
#include "launcher.h"
#include "shared_array.h"
#include "command_list.h"
#include "TargetConfig.h"

template<typename T>
using Buffer = SharedArray<T>;
// using Buffer = std::vector<T>;


int main()
{
    std::cout << "Hello, Asynchronous Waiting!" << std::endl;
    
    // Init metal system
    {
        create_device();

        init_command_list();
    }

    gpuFunction fn_test_add_gpu;

    auto fn_test_add_cpu = [](Buffer<uint>& input_ptr, const uint desire_value)
    {
        // fast_format("CPU get {} (desire for {})", input_ptr[0], desire_value);
        if (input_ptr[0] == desire_value) { input_ptr[0] += 1; }
    };

    //
    // Load Functions
    //
    {
        std::string full_path_xpbd = std::string(SELF_RESOURCES_PATH) + std::string("/metal_libs/") + std::string("example3.metallib");
        NS::Error* err; MTL::Library* library_xpbd = get_device() -> newLibrary(std_string_to_ns_string(full_path_xpbd), &err);
        check_err(library_xpbd, err);
        
        fn_test_add_gpu.load(library_xpbd, "test_add_1");
    }

    Buffer<uint> test_buffer;
    test_buffer.resize(1); test_buffer.set_data(0);

    const uint loop_time = 4;

    //
    // Test async waiting: Add single value for loop_time times, Odd on the CPU, Even on the GPU
    //   GPU ->     1     3   5    7   9
    //   CPU ->  0     2    4   6    8
    // 
    {
        get_shared_event().refresh();
        get_command_list().reset_auto_fence_count();
        
        //
        // Launch GPU Commands : Async
        //
        const uint num_gpu_events = loop_time;

        std::vector< MTL::CommandBuffer* > list_cmd_buffer(num_gpu_events);

        for (uint cmd_idx = 0; cmd_idx < num_gpu_events; cmd_idx++) 
        {
            const uint curr_event = 2 * cmd_idx + 1;
            const uint prev_cpu_event = curr_event - 1;

            // fast_print(cmd_idx, "GPU Signal", cmd_idx);
            list_cmd_buffer[cmd_idx] = get_command_list().start_new_list_with_new_buffer();

            // Wait for CPU's signal
            {
                // fast_print(cmd_idx, "GPU Waits for CPU", prev_cpu_event);
                get_command_list().wait_cpu(get_shared_event(), prev_cpu_event);     
            }
            
            // Launch
            {
                // fast_format("GPU desire for {}", curr_event);
                get_command_list().add_task(fn_test_add_gpu);
                fn_test_add_gpu.bind_ptr(test_buffer);
                fn_test_add_gpu.bind_constant(curr_event);
                fn_test_add_gpu.launch_async(1);
            }   
            get_command_list().make_fence_with_previous_cmd_buffer(); // If False, The Function May Be Empty
            get_command_list().send_last_cmd_buffer_in_list();
        }

        ///
        /// Launch CPU Commands : Immediate
        ///
        const uint num_cpu_events = loop_time;
        
        std::vector<float> runtime_cost_cpu(num_cpu_events, 0.0f);
        for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) 
        {
            const uint curr_event = 2 * cmd_idx;
            const uint prev_gpu_event = curr_event - 1;
            const uint prev_gpu_idx = prev_gpu_event / 2;
            const uint next_gpu_event = curr_event + 1;

            // Wait for GPU's fence
            {
                if (curr_event != 0)
                {
                    // fast_print(cmd_idx, "CPU Waits for GPU", prev_gpu_idx);
                    list_cmd_buffer[prev_gpu_idx]->waitUntilCompleted();
                }
            }
            
            // Launch
            // fast_format("CPU desire for {}", curr_event);
            fn_test_add_cpu(test_buffer, curr_event);

            // Signal
            {
                // fast_print(cmd_idx, "CPU Signal", curr_event);
                get_shared_event().event->setSignaledValue(curr_event);
            }
        }

        get_command_list().wait_all_cmd_buffers();
    }  

    fast_format("Final value = {} (desire for {})", test_buffer[0], loop_time * 2);


    return 0;
}