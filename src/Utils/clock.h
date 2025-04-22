#pragma once

#include "scalar.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <thread>


#ifdef _WIN32
#include <winsock.h>
#elif __APPLE__
#include <sys/time.h>
// #include <sys/_types/_timeval.h>
#endif

class SimClock{

private:
    using Cost = double;
    using FunctionDuration = std::pair<std::string, timeval>;
    
    std::vector<FunctionDuration> func_duration_vec;
    timeval start_timeval;
    std::vector<Cost> time_cost_vec = std::vector<Cost>();
    bool is_timer_output_active = true;

    long ms, sc;
    Cost clock_result;

public:

    void enable_timer();
    void disable_timer();
    bool is_timer_active();
    void clear_timer_list();

    // 在函数开始前调用，记录函数开始时的时间戳
    void add_time_stamp(std::string func_name) ;

    void start_clock();
    Cost end_clock();
    Cost duration();

    // 输出所有的时间
    void output_func_duration();

    void output_timeline_duration();

    // 记录最后的时间戳，输出之前所有的间隔时间
    void end_clock_and_clear_and_output();


    void output_timeline_graph();

    static void sleep_ms(Cost t);

    template<typename Func>
    static double loop_for(Func function, uint loop_count = 100)
    {
        SimClock clock; clock.start_clock();
        for (int i = 0; i < 100; i++) { function(); }
        return clock.end_clock();
    }
    template<uint Count>
    static void analysic_breakdown(const std::vector<double>& list_cost, const std::vector<std::string>& list_name)
    {
        double sum = 0;
        for(const double dt : list_cost) { sum += dt; }
        std::cout << "Cost Detail : ";
        for (const double dt : list_cost) {  std::cout << std::format("{:.4f}  / ", dt/((float)Count)); }
        std::cout << std::endl;
        
        std::cout << "Breakdown   : ";
        for (const double dt : list_cost) {  std::cout << std::format("{:2.4f}% / ", dt / sum * 100); }
        std::cout << std::endl;

        if(!list_name.empty()){
            std::cout << "(In         : ";
            for (const auto name : list_name) {  std::cout << name << " / "; }
            std::cout << std::endl;
        }
    }

};