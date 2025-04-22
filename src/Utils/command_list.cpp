#include "command_list.h"

CommandList command_list[num_launcher];
sim::Fence fence_list[num_fence];
sim::Event event_list[num_fence];
sim::SharedEvent shared_event_list[num_fence];

void init_command_list(){
    for (uint i = 0; i < num_launcher; i++) {
        command_list[i].set_up_device();
    }
    for (uint i = 0; i < num_fence; i++) {
        #if defined (__APPLE__)
        fence_list[i].fence = get_device() -> newFence();
        event_list[i].event = get_device() -> newEvent();
        shared_event_list[i].event = get_device() -> newSharedEvent();
        #endif
    }
}

CommandList& get_command_list(uint idx){
    return command_list[idx];
}

gpuFunction& add_compute_task(gpuFunction& func, uint queue_idx){
    #if defined (__APPLE__)
    get_command_list(queue_idx).add_task(func);
    #endif
    return func;
}

sim::Fence& get_fence(uint idx){
    return fence_list[idx%num_fence];
}
sim::Event& get_event(uint idx){
    return event_list[idx%num_fence];
}
sim::SharedEvent& get_shared_event(uint idx){
    return shared_event_list[idx%num_fence];
}