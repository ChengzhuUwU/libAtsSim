#pragma once

///
/// GPU Fence & Event For Data Sync
///

#include "bits_utils.h"
#include <metal_system.h>

namespace sim
{
class Fence{
public:
#if defined (__APPLE__)
    MTL::Fence* fence;
#else
    uint* fence;
#endif
public:
    inline void refresh(){
        #if defined (__APPLE__)
        fence = get_device()->newFence();
        #endif
    }
};

class Event{
public:
#if defined (__APPLE__)
    MTL::Event* event;
#else
    uint* event;
#endif

public:
    
};


class SharedEvent{
public:
#if defined (__APPLE__)
    MTL::SharedEvent* event;
#else
    uint* event;
#endif

public:
    inline void mark_event_done(const uint64_t& value){
        #if defined (__APPLE__)
        event->setSignaledValue(value);
        #endif
    }
    inline void refresh(){
        #if defined (__APPLE__)
        event = get_device()->newSharedEvent();
        #endif
    }
};

// class CommandBuffer{
// public:
// #if defined (__APPLE__)
//     MTL::CommandBuffer* buffer;
// #else
//     uint* buffer;
// #endif

// public:
//     void send(){
//         #if defined (__APPLE__)
//         buffer->commit();
//         #endif
//     }
//     void wait(){
//         #if defined (__APPLE__)
//         buffer->waitUntilCompleted();
//         #endif
//     }
    
// };

}


