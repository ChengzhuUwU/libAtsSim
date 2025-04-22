#include "clock.h"

// class SimClock {


// timeval prev_time_stamp;


using Cost = double;

#ifdef WIN32
static inline int gettimeofday(timeval* tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year  = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday  = wtm.wDay;
  tm.tm_hour  = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -1;
  clock = mktime(&tm);
  tp -> tv_sec = clock;
  tp -> tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif

void SimClock::enable_timer()      { is_timer_output_active = true;  }
void SimClock::disable_timer()     { is_timer_output_active = false; }
bool SimClock::is_timer_active()   { return is_timer_output_active;  }
void SimClock::clear_timer_list()  { func_duration_vec.clear();      }

// 在函数开始前调用，记录函数开始时的时间戳
void SimClock::add_time_stamp(std::string func_name) {
    
    if(!func_duration_vec.empty() && func_name == func_duration_vec[0].first){
        // 如果忘了手动清零，则在每帧第一次计时的时候清零
        clear_timer_list();
    }
        
    timeval current_time_stamp;
    gettimeofday(&current_time_stamp, nullptr);
    func_duration_vec.push_back(std::make_pair(func_name, current_time_stamp)); 

    // 更新上一个函数时间   
    // prev_time_stamp = current_time_stamp;
}

void SimClock::start_clock(){
    // clear_timer_list();
    // add_time_stamp("start");
    gettimeofday(&start_timeval, nullptr);
}

Cost SimClock::end_clock(){
    // add_time_stamp("end");

    timeval end_timeval;
    gettimeofday(&end_timeval, nullptr);

    // Cost result;
    {  
        // auto start_stamp = func_duration_vec[0];
        // auto end_stamp = func_duration_vec[func_duration_vec.size() - 1];
        auto& start_stamp = start_timeval;
        auto& end_stamp = end_timeval;

        auto microseconds = end_stamp.tv_usec - start_stamp.tv_usec;
        auto seconds = end_stamp.tv_sec - start_stamp.tv_sec;

        // sc = seconds;
        // ms = microseconds;
        // time_cost_vec.push_back(macro_duration);

        // if (return_second)  { Cost second_duration = microseconds / 1000000.0 + seconds;        return second_duration; }
        // else                
        clock_result = microseconds / 1000.0 + seconds * 1000.0;  
        return clock_result;
    }
    // clock_result = result;
    // return result;
}
Cost SimClock::duration(){
    // Cost second_duration = ms / 1000000.0 + sc; // return second
    // Cost macro_duration = ms / 1000.0 + sc * 1000.0;
    // time_cost_vec.push_back(macro_duration);
    // if(return_second) return second_duration;
    // else 
    return clock_result;
}

// 输出所有的时间
void SimClock::output_func_duration(){
    // if(is_timer_output_active){
    //     std::cout << " -- Time Cost of Each Function  --  \n";
    //     Cost total_cost = 0;
    //     for (int i = 0; i < func_duration_vec.size() - 1; i++) {
    //         func_duration current_stamp = func_duration_vec[i];
    //         auto next_stamp = func_duration_vec[i + 1];

    //         auto seconds = next_stamp.second.tv_sec - current_stamp.second.tv_sec;
    //         auto microseconds = next_stamp.second.tv_usec - current_stamp.second.tv_usec;
    //         auto duration = seconds * 1000000.0 + microseconds;
    //         // auto duration = microseconds;
    //         Cost float_duration = duration / 1000.0; 
    //         total_cost += float_duration; 

    //         std::cout << "( " << current_stamp.first << " : " << float_duration << " ms) " << std::endl;
    //     }
    //     std::cout << "( TOTAL COST : " << total_cost << " ms ) " << std::endl;
    //     // std::cout << " -- total cost : " << total_cost << " ms -- " << std::endl;
    // }
}

void SimClock::output_timeline_duration(){
    std::cout << " ------- Time Cost of Timeline (ms)  -------  \n [ ";
    for (int i = 0; i < time_cost_vec.size() - 1; i++) {
        std::cout << time_cost_vec[i] << ", ";
    }
    std::cout << time_cost_vec[time_cost_vec.size() - 1] << " ] \n";
    time_cost_vec.clear();
}

// 记录最后的时间戳，输出之前所有的间隔时间
void SimClock::end_clock_and_clear_and_output(){
    end_clock();
    output_func_duration();
    clear_timer_list();
}




const int graph_height = 20;


void SimClock::output_timeline_graph(){
    std::string graph[100][graph_height];
    std::cout << " ------- Cost Graph of Timeline (ms)  -------  \n";
    uint graph_length = static_cast<uint>(min_scalar(time_cost_vec.size(), 100ull));
    uint start_pos = static_cast<uint>(max_scalar(0ull, time_cost_vec.size() - 100));
    Cost max_cost = fast_max(time_cost_vec);

    for (int i = 0; i < graph_height; i++) {
        for (int j = 0; j < graph_length; j++) {
            graph[i][j] = " ";
        }
    }
    
    int j = 0;
    for (int i = start_pos; i < time_cost_vec.size(); i++, j++) {
        int height = graph_height * (time_cost_vec[i] / max_cost);
        fast_print(height);
        for(int n = graph_height - 1; n >= graph_height - height; n--){
            graph[j][n] = "=";
        }
    }

    for (int i = 0; i < graph_height; i++) {
        for (int j = 0; j < graph_length; j++) {
            std::cout << graph[j][i];
            // std::cout << "=";
        }
        std::cout << "\n";
    }

    time_cost_vec.clear();
}

void SimClock::sleep_ms(Cost t){
    uint micro_time = t * 1000.0;
    std::this_thread::sleep_for(std::chrono::microseconds(micro_time));
    // std::this_thread::sleep_for(std::chrono::nanoseconds());
}



// };
