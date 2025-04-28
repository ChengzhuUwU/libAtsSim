#if __APPLE__
    #ifndef NS_PRIVATE_IMPLEMENTATION
        #define NS_PRIVATE_IMPLEMENTATION
        #define CA_PRIVATE_IMPLEMENTATION
        #define MTL_PRIVATE_IMPLEMENTATION
    #endif
#endif

#include "launcher.h"
#include "clock.h"
#include "command_list.h"
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>



namespace Launcher {



template <typename T, typename FuncGetSuccNodes>
inline bool visit_dag_dfs(const uint node, const std::vector<T>& tasks, std::vector<uint>& visited, std::stack<uint>& result, FuncGetSuccNodes func_get_succ_nodes) {
    visited[node] = 1;
    const auto& successors = func_get_succ_nodes(tasks[node]);
    for (const uint successor : successors) {
        if (visited[successor] == 0) {
            const bool succ = visit_dag_dfs(successor, tasks, visited, result, func_get_succ_nodes);
            if (!succ) {
                return false;
            }
        }
        else if (visited[successor] == 1)  {
            return false;
        }
    }
    visited[node] = 2;
    result.push(node);
    return true;
}
template <typename T, typename FuncGetSuccNodes>
inline std::vector<uint> topological_sort_DFS(const std::vector<T>& tasks, FuncGetSuccNodes func_get_succ_nodes) {
    std::vector<uint> visited(tasks.size(), 0);
    std::stack<uint> stack;
    std::vector<uint> invalid_nodes;
    for (uint tid = 0; tid < tasks.size(); tid++) {
        if (!visited[tid]) {
            const bool succ = visit_dag_dfs(tid, tasks, visited, stack, func_get_succ_nodes);
            if (!succ) {
                invalid_nodes.push_back(tid);
            }
        }
    }

    if (invalid_nodes.empty()) {
        std::vector<uint> result;
        while (!stack.empty()) {
            result.push_back(stack.top());
            stack.pop();
        }
        return result;
    }
    else {
        return invalid_nodes;
    }
    
};
template <typename T, typename FuncGetSuccNodes>
inline std::vector<uint> topological_sort_BFS(const std::vector<T>& tasks, FuncGetSuccNodes func_get_succ_nodes) {
    uint num_tasks = tasks.size();
    std::vector<uint> inDegree(num_tasks, 0);
    std::vector<uint> result;

    for (const auto& task : tasks) {
        const auto& successors = func_get_succ_nodes(task);
        for (const auto& successor : successors) {
            inDegree[successor]++;
        }
    }
    std::queue<int> q;
    for (uint i = 0; i < num_tasks; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        const auto& task = tasks[node];
        const auto& successors = func_get_succ_nodes(task);
        for (auto successor : successors) {
            inDegree[successor]--;
            if (inDegree[successor] == 0) {
                q.push(successor);
            }
        }
    }
    if (result.size() == tasks.size()) {
        return result;
    }
    else {
        std::vector<uint> invalid_nodes;
        std::set<uint> valid_set(invalid_nodes.begin(), invalid_nodes.end());
        for (uint i = 0; i < num_tasks; i++) {
            if (valid_set.find(i) == valid_set.end()) {
                invalid_nodes.push_back(i);
            }
        }
        return invalid_nodes;
    }
};

inline std::vector<uint> topological_sort_DFS(const std::vector<Task>& tasks) {
    return topological_sort_DFS(tasks, [](const Task& task) { return task.successors; });
};
inline std::vector<uint> topological_sort_DFS(const std::vector<MergedTask>& tasks) {
    return topological_sort_DFS(tasks, [](const MergedTask& task) { return task.successors; });
};
inline std::vector<uint> topological_sort_BFS(const std::vector<Task>& tasks) {
    return topological_sort_BFS(tasks, [](const Task& task) { return task.successors; });
};
inline std::vector<uint> topological_sort_BFS(const std::vector<MergedTask>& tasks) {
    return topological_sort_BFS(tasks, [](const MergedTask& task) { return task.successors; });
};



///
/// DAG
///
bool Scheduler::topological_sort(){
    
    // if (!constraint_task_orders.empty())
    // {
    //     std::vector<Task> enhanced_list_task(list_task);
    //     for (uint i = 1; i < constraint_task_orders.size(); i++)
    //     {
    //         const uint prev_node = constraint_task_orders[i - 1];
    //         const uint curr_node = constraint_task_orders[i];
    //         enhanced_list_task[prev_node].add_front(curr_node);
    //         enhanced_list_task[curr_node].add_back(prev_node);
    //     }
    //     list_order = topological_sort_DFS(enhanced_list_task); 
    // }
    // else 
    {
        list_order = topological_sort_DFS(list_task); 
        // list_order = topological_sort_BFS(list_task); 
    }

    // print_sort_by_typology();

    // fast_print("Registed Task Num", list_task.size(), "Sorted Task Num", list_order.size());
    // std::vector<Task> tmp_task_list(list_task);
    // for(uint i = 0; i < list_task.size(); i++){
    //     auto& tid = list_order[i];
    //     list_task[i] = tmp_task_list[tid];
    // }
    // for(uint i = 0; i < list_task.size(); i++){
    //     list_order[i] = i;
    // }

    // 如果结果长度小于节点数,说明存在环路
    if (list_order.size() != list_task.size()) {
        fast_format_err("\n\n   Error : Graph Constains A Cycle");
        for (const uint invalid_node : list_order) {
            fast_format_err("   Invalid Node {:2} ({})", invalid_node, taskNames.at(list_task[invalid_node].func_id));
        }
        fast_format_err("\n\n");
        list_order.clear();
        return false;
    }
    else {
        // standardizing_dag();

        //
        // Post Processing : Disperse The Distribution of RankU 
        //


        return true;
    }
    
}

void Scheduler::set_loop_count(const uint loop_count){
    ///
    /// Has to called profile()
    ////
    const uint num_tasks = list_task.size();
    uint current_prefix = num_tasks;
    
    for (uint i = 1; i < loop_count; i++) {
        // fast_print("num_tasks", num_tasks);
        // fast_print("current_prefix", current_prefix);
        for (uint tid = 0; tid < num_tasks; tid++) {
            const Task& task = list_task[tid];
            Task new_task(task);
            new_task.predecessors.clear(); new_task.successors.clear();
            uint new_tid = add_task(new_task);
            computation_matrix.push_back(computation_matrix[tid]); 
        }
        for (uint tid = 0; tid < num_tasks; tid++) {
            const Task& task = list_task[tid];
            bool is_root = tid == root_node;
            bool is_terminal = tid == terminal_node;
            uint new_tid = tid + current_prefix;
            for (uint pred : task.predecessors) { set_connect(current_prefix + pred, new_tid); }
            /// Should NOT Copy Succ Again, Otherwise Connect Will be Redundant
            // for (uint succ : task.successors)   { set_connect(new_tid, current_prefix + succ); }
            if(is_root) { uint prev_prefix = current_prefix - num_tasks; set_connect(prev_prefix + terminal_node, new_tid); }
            if(is_terminal && i == loop_count - 1) { terminal_node = new_tid; }
        }
        current_prefix += num_tasks;
    }
    // print_tasks();
    /// , const uint start_idx, const uint end_idx
}

void Scheduler::print_sort_by_typology(){
    if (!list_order.empty()) {
        std::cout << " \n ----------------------- \n  Topological order:\n";
        for (int tid : list_order) {
            list_task[tid].print_with_cluster(tid);
            // std::cout << "    " << Launcher::taskNames.at(list_task[tid].func_id) << "\n";
        }
        std::cout << std::endl;
    }
}
void Scheduler::print_sort_by_ranku(){
    const bool print_ranku = true;
    if (print_ranku) {
        // fast_print("Sort by Topology Sort");
        // for (auto tid : list_order) {
        //     auto& task = list_task[tid];
        //     fast_format(" {:6.3f} : [Iter {:2}, Color {:2}] => {}", ranku[tid], task.iter_idx, task.cluster_idx, Launcher::taskNames.at(task.func_id) );
        // }
        fast_print("Sort by Ranku");
        for (auto tid : sorted_nodes) {
            auto& task = list_task[tid];
            fast_format(" {:6.3f} : [Tid {:2}, Iter {:2}, Color {:2}] => {}", ranku[tid], tid, task.iter_idx, task.cluster_idx, Launcher::taskNames.at(task.func_id) );
            // std::cout << "  " << ranku[tid] << " : " << "[" << task.cluster_idx << "] " << Launcher::taskNames.at(task.func_id) << " \n";
            // std::cout << "node : " << tid << " , rank u = " << ranku[tid] << " \n";
        }
    }  

    if constexpr (false)
    {
        std::map<FunctionID, uint> funcid_color_map = {
            {id_xpbd_predict_position, 1},
            {id_self_collision_update_vert_aabb_dcd, 2},
            {id_obstacle_collision_update_face_aabb_dcd, 3},
            {id_self_collision_apply_leaves_aabb, 4},
            {id_obstacle_collision_apply_leaves_aabb, 5},
            {id_self_collision_query_from_cloth_vert, 6},
            {id_obstacle_collision_query_from_cloth_vert, 7},
            {id_self_collision_reset_collision_system_and_lbvh, 8},
            {id_obstacle_collision_reset_collision_system_and_lbvh, 9},
            {id_self_collision_narrow_phase_vv, 10},
            {id_obstacle_collision_narrow_phase_vf, 11},
            {id_graph_coloring_vivace_vv_cloth, 12},
            {id_graph_coloring_vivace_vv_tet, 12},
            {id_xpbd_reset_constrains, 13},

            {id_xpbd_constraint_stretch_mass_spring_half, 14},
            {id_xpbd_constraint_bending_half, 15},
            {id_xpbd_constraint_self_collision_vv_half_cloth, 16},
            {id_xpbd_constraint_obstacle_collision_vv_cloth, 17},
        };

        // auto order_dfs = topological_sort_DFS(list_task);
        // auto order_bfs = topological_sort_BFS(list_task);
        fast_print("BFS = [ ");
        for (const uint tid : sorted_nodes)
        {
            auto task = list_task[tid];
            if (funcid_color_map.find(task.func_id) != funcid_color_map.end())
            {
                uint mapped_id = funcid_color_map[task.func_id];
                uint cluster_id = mapped_id >= 14 && mapped_id <= 18 ? task.cluster_idx + 1 : 0;
                fast_format_single("({}, {}, {:6.3f}), ", mapped_id, cluster_id, ranku[tid]);
            }
        }
        fast_print(" \n]");
    }
    
   

}

// struct FunctionImplementation{
//     uint imp_id;
//     FunctionID id;
//     FunctionImplementation(const Implementation& implement, const FunctionID& func_id) :
//         imp_id(implement.implementation_id), id(func_id) {}
//     bool operator == (const FunctionImplementation& right) const {
//         return imp_id == right.imp_id && id == right.id;
//     }
//     friend bool operator==(CREF(FunctionImplementation) left, CREF(FunctionImplementation) right){
//         return left.imp_id == right.imp_id && left.id == right.id;
//     }
// };  

void Scheduler::profile(const std::function<void()>& restart_system, const std::function<LaunchParam(const Task&)> task_to_param, const std::vector< std::vector<float> >& profiled_comp_matrix){

    //  const uint max_loop_count = 53;
     const uint max_loop_count = 23;

    // Assume that each function has only 1 cost on single device
    //  std::unordered_map<FunctionImplementation, float> cost_map;

    // std::vector<std::pair<uint, FunctionID>> imp_map;
    // std::vector<float> cost_map;
    // auto fn_get_cost = [&imp_map, &cost_map](const Implementation& implement, const FunctionID& func_id) -> double{
    //     for (uint i = 0; i < cost_map.size(); i++) {
    //         const auto& pair = imp_map[i];
    //         if (pair.first == implement.implementation_id && pair.second == func_id) {
    //             return cost_map[i];
    //         }
    //     }
    //     return -1.0;
    // };
    // auto fn_insert_cost = [&imp_map, &cost_map](const Implementation& implement, const FunctionID& func_id, const float cost) -> void{
    //     imp_map.push_back(std::make_pair(implement.implementation_id, func_id));
    //     cost_map.push_back(cost);
    // };

    if(profiled_comp_matrix.empty()){

        fast_print("Begin Profile CPU");

        const uint num_tasks = list_task.size();
        std::vector< std::vector<double> > cost_list_cpu(num_tasks);
        std::vector< std::vector<double> > cost_list_gpu(num_tasks);
       
        /// Profile CPU
        // fast_print("Profile CPU");
        for (uint loop = 0; loop < max_loop_count; loop++) {

            fast_print("Loop Count", loop);

            for (auto tid : list_order) {
                auto& task = list_task[tid];
                bool find;
                const auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if(find){
                    double dt;
                    // double geted_cost = fn_get_cost(imp, task.func_id);
                    // if (geted_cost == -1.0) 
                    // {
                        SimClock clock;
                        clock.start_clock();

                        // imp.launch_task({task.func_id, 0, task.num_threads, 256, 0, 0, true, true, false});
                        imp.launch_task(task_to_param(task));

                        dt = clock.end_clock();
                    //     fn_insert_cost(imp, task.func_id, dt);
                    // }
                    // else {
                    //     dt = geted_cost;
                    // }
                    
                    if(loop != 0) cost_list_cpu[tid].push_back(dt);
                }
            }
            restart_system();
        }

        // fast_print("Begin Profile GPU");
        
        #if defined (__APPLE__)
        /// Profile GPU
        fast_print("Profile GPU");
        for (uint loop = 0; loop < max_loop_count; loop++) {

            fast_print("Loop Count", loop);

            // if(loop == 0) get_scene_params().print_task_info = true;
            // if(loop == 1) get_scene_params().print_task_info = false;
            // std::vector<MTL::CommandBuffer*> list_cmd(list_task.size());

            std::vector<std::pair<MTL::CommandBuffer*, MTL::CommandBuffer*>> list_cmd_buffer(list_order.size());

            for (uint i = 0; i < list_order.size(); i++) {
                auto tid = list_order[i];
                auto& task = list_task[tid];
                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                if(find){
                    // double geted_cost = fn_get_cost(imp, task.func_id);
                    // if (geted_cost == -1.0) {
                    //     if (task.is_empty_node) {
                    //         dt = 0.0;
                    //     }
                    //     else 
                    //     {

                    get_command_list().start_new_list();
                    MTL::CommandBuffer* begin_buffer = get_command_list().command_buffer;

                    // imp.launch_task({task.func_id, 0, task.num_threads, 256, 0, 0, true, true, false});
                    imp.launch_task(task_to_param(task));

                    get_command_list().send_list();
                    MTL::CommandBuffer* end_buffer = get_command_list().command_buffer;

                    list_cmd_buffer[i] = std::make_pair(begin_buffer, end_buffer);

                    // double dt = (end_buffer->GPUEndTime() - begin_buffer->GPUStartTime()) * 1000.0;
                    // if(loop != 0) { cost_list_gpu[tid].push_back(dt); }

                }
                else{
                    // if(loop == 0){
                    //     fast_print("Empty GPU Implementation", Launcher::taskNames.at(task.func_id));
                    // }

                    bool find_cpu;
                    auto& imp_cpu = task.get_implementation(Launcher::DeviceTypeCpu, find);
                    if(find){
                        get_command_list().wait();
                        
                        SimClock clock; clock.start_clock();

                        imp_cpu.launch_task(task_to_param(task));

                        double dt = clock.end_clock();

                        if(loop != 0) { cost_list_gpu[tid].push_back(dt); }
                    }
                    else{
                        std::cerr << " -------- Task " << Launcher::taskNames.at(task.func_id) << " does not has any implementation --------- \n";
                    }
                }
            }

            get_command_list().wait();
            
            for (uint i = 0; i < list_order.size(); i++){
                const auto& pair = list_cmd_buffer[i];

                auto tid = list_order[i];
                auto& task = list_task[tid];
                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                if (find) {
                    double dt = (pair.second->GPUEndTime() - pair.first->GPUStartTime()) * 1000.0;
                    if(loop != 0) { cost_list_gpu[tid].push_back(dt); }
                }

            }

            // get_command_list().send_and_wait();

            restart_system();
        }
        #endif

        double sum_gpu = 0.0;
        double sum_cpu = 0.0;

        /// Compute Speed
        computation_matrix.resize(num_tasks);
        for (uint tid = 0; tid < list_task.size(); tid++) {
            auto& cost_cpu = cost_list_cpu[tid];
            auto& cost_gpu = cost_list_gpu[tid];
            auto& task = list_task[tid];
            auto& comp_mat = computation_matrix[tid];
            comp_mat = {inf, inf};

            // fast_print_iterator(cost_cpu);

            if(!cost_cpu.empty()){
                std::sort(cost_cpu.begin(), cost_cpu.end());
                float total_cpu = 0.f;
                // std::cout << taskIdx << " cpu ," << Launcher::taskNames.at(task.func_id) << " : ";  
                for (uint i = 2; i < cost_cpu.size() - 2; i++) {
                    float dt = cost_cpu[i];
                    total_cpu += dt;
                    // fast_print_single(dt);
                }
                // fast_print("");
                double avg_dt = total_cpu / double(cost_cpu.size() - 4);
                sum_cpu += avg_dt;
                comp_mat[0] = float(avg_dt);
            }

            if(!cost_gpu.empty()) {
                std::sort(cost_gpu.begin(), cost_gpu.end());    
                float total_gpu = 0.f;
                // std::cout << taskIdx << " gpu , " << Launcher::taskNames.at(task.func_id) << " : ";  
                for (uint i = 2; i < cost_gpu.size() - 2; i++) {
                    float dt = cost_gpu[i];
                    total_gpu += dt;
                    // fast_print_single(dt);
                }
                // fast_print("");
                float avg_dt = total_gpu / float(cost_gpu.size() - 4);
                comp_mat[1] = (avg_dt);
                sum_gpu += avg_dt;
            }
            else{
                if(!cost_cpu.empty())
                    sum_gpu += comp_mat[0];
                comp_mat[1] = (inf);
            }

        }

        fast_format("Homogeneous Environment Cost in CPU = {} , GPU = {}", sum_cpu, sum_gpu);

        /// Output Information
        const bool print_task_matrix = false; /// --- Print --- 
        if(print_task_matrix){
            const uint num_tasks = list_task.size();
            fast_print("Task Lists");
            for (uint tid = 0; tid < num_tasks; tid++){
                std::cout << "  " << tid << " : " << taskNames.at(list_task[tid].func_id) << std::endl;
            }
            fast_print("Cost Matrix");
            std::cout << num_tasks << " " << 2 << std::endl;
            std::vector<double> sum_t(2, 0.0);
            for (uint tid = 0; tid < num_tasks; tid++) {
                auto& task = list_task[tid];
                const auto& comp_mat = computation_matrix[tid];
                float cost_cpu = comp_mat[0];
                float cost_gpu = comp_mat[1];
                sum_t[0] += cost_cpu;
                if(cost_gpu != inf) sum_t[1] += cost_gpu;
                else                sum_t[0] += cost_cpu;
                std::cout << "    {" << cost_cpu << ", " << cost_gpu << "},\n";
            }
            fast_print_single("Total Costs : "); fast_print("CPU", sum_t[0], "GPU", sum_t[1]);
        }
    }
    else{

        // fast_print(profiled_comp_matrix.size());

        if(profiled_comp_matrix.size() == list_task.size()){
            computation_matrix = profiled_comp_matrix;
        }
        else{
            fast_print_err("Size of the Profiled Computation Matrix Does NOT Consistent With Curreng DAG");
        }
        
    }
    
    // if(print_task_matrix) std::cout << 0.f << " " << 1.f << "\n";
    // if(print_task_matrix) std::cout << 1.f << " " << 0.f << "\n";

    orig_list_task = list_task;
    orig_computation_matrix = computation_matrix;

}

void Scheduler::profile(
    std::function<void(
        const std::vector<uint>& list_order, 
        const std::vector<Task>& list_task, 
        std::vector< std::vector<double> >& cost_list_cpu,
        std::vector< std::vector<double> >& cost_list_gpu,
        std::vector<double> & cost_total
        )> func_get_costs)
{
    const uint num_tasks = list_task.size();
    std::vector< std::vector<double> > cost_list_cpu(num_tasks);
    std::vector< std::vector<double> > cost_list_gpu(num_tasks);
    std::vector< double > cost_total;

    func_get_costs(list_order, list_task, cost_list_cpu, cost_list_gpu, cost_total);

    double sum_gpu = 0.0;
    double sum_cpu = 0.0;

    /// Compute Speed
    computation_matrix.resize(num_tasks);

    for (uint tid = 0; tid < list_task.size(); tid++) 
    {
        auto& cost_cpu = cost_list_cpu[tid];
        auto& cost_gpu = cost_list_gpu[tid];
        auto& task = list_task[tid];
        auto& comp_mat = computation_matrix[tid];
        comp_mat = {inf, inf};

        const uint drop_count = 4;
        // CPU Part
        if (!cost_cpu.empty() && task.has_implementation(DeviceTypeCpu))
        {
            std::sort(cost_cpu.begin(), cost_cpu.end());
            double total_cpu = std::accumulate(cost_cpu.begin() + drop_count, cost_cpu.end() - drop_count, 0.0);
            double avg_dt = total_cpu / double(cost_cpu.size() - drop_count * 2);
            sum_cpu += avg_dt;
            
            comp_mat[0] = float(avg_dt);
        }
        else { fast_format("Does Not Has CPU Implementation {}", taskNames.at(task.func_id)); }

        // GPU Part
        if (!cost_gpu.empty() && task.has_implementation(DeviceTypeGpu)) 
        {
            std::sort(cost_gpu.begin(), cost_gpu.end());    
            double total_gpu = std::accumulate(cost_gpu.begin() + drop_count, cost_gpu.end() - drop_count, 0.0);
            double avg_dt = total_gpu / double(cost_gpu.size() - drop_count * 2);
            sum_gpu += avg_dt;

            comp_mat[1] = (avg_dt);
        }
        else 
        {
            if (!cost_cpu.empty()) sum_gpu += comp_mat[0];
            comp_mat[1] = (inf);
        }

    }


    fast_print("\n======================= New Scheduler ===========================");
    fast_print("total cpu", cost_total[0], "sum of task", sum_cpu);
    fast_print("total gpu", cost_total[1], "sum of task", sum_gpu);
    // cost_total[0] = sum_cpu;

    communication_cost_matrix_uma[0][0] = (cost_total[0] - sum_cpu) / double(num_tasks);
    communication_cost_matrix_uma[1][1] = (cost_total[1] - sum_gpu) / double(num_tasks);
    if (communication_cost_matrix_uma[0][0] < 0 || communication_cost_matrix_uma[1][1] < 0) fast_format_err("Communication Time Is Less Than 0, May Lossing The Connection");
    fast_print("Communication Cost Inner CPU", communication_cost_matrix_uma[0][0], "Inner GPU", communication_cost_matrix_uma[1][1]);

    orig_list_task = list_task;
    orig_computation_matrix = computation_matrix;
    summary_of_costs_each_device = cost_total;

}

// Reading From Precomputed CostMap
void Scheduler::profile_from(
        const std::vector< std::pair<Launcher::FunctionID, uint> >& map_tasks, 
        const std::vector< std::vector<double> >& map_costs,
        std::vector<double> & cost_total)
{
    auto fn_get_cost_template = [&map_tasks, &map_costs](const Launcher::FunctionID& func_id, const uint& cluster_idx) -> std::vector<double>
    {
        for (uint i = 0; i < map_tasks.size(); i++) 
        {
            const auto& pair = map_tasks[i];
            if (pair.first == func_id && pair.second == cluster_idx) 
            {
                // fast_format("Get Cost of {} / {} : {} & {}", taskNames.at(func_id), cluster_idx, map_costs[i][0], map_costs[i][1]);
                return map_costs[i];
            }
        }
        fast_format_err("Can NOT Find Key of Task {} / {}", taskNames.at(func_id), cluster_idx);
        return std::vector<double>{0, 0};
    };

    const uint num_tasks = list_task.size();
    const uint num_procs = map_costs.back().size();
    std::vector<double> cost_sum(num_procs, 0.0);

    /// Compute Speed
    computation_matrix.resize(num_tasks);

    for (uint tid = 0; tid < list_task.size(); tid++) 
    {
        auto& task = list_task[tid];
        auto& comp_mat = computation_matrix[tid];
        comp_mat.resize(num_procs, inf);

        const auto& costs = fn_get_cost_template(task.func_id, task.cluster_idx);

        for (uint proc = 0; proc < num_procs; proc++)
        {
            float curr_dt = costs[proc];
            comp_mat[proc] = task.has_implementation(proc) ? float(curr_dt) : inf;
            cost_sum[proc] += costs[proc];
            if (costs[proc] < 0) { fast_print_err("Cost Is Less Than 0", taskNames.at(task.func_id)); return; }
        }
    }

    
    // fast_print("\n======================= New Scheduler ===========================");
    // for (uint proc = 0; proc < num_procs; proc++)
    // {
    //     fast_format("Total Cost of {} : {}", proc, cost_sum[proc]);
    // }

    orig_list_task = list_task;
    orig_computation_matrix = computation_matrix;
    if (cost_total.empty())
    {
        cost_total = cost_sum;
    }
    summary_of_costs_each_device = cost_total;
    
}

// If use 'LaunchModeHetero', we do not need do sync after this function
void Scheduler::launch(LaunchMode mode, const std::function<LaunchParam(const Task&)> task_to_param, const bool fully_not_wait, const std::vector<std::function<void()>>& assemble_impl)
{

    #if __APPLE__
    //
    // 简单串行
    //
    if (mode == LaunchModeGpu) 
    {
        // list_order
        if (sorted_nodes.empty()) { fast_print_err("List Order Is EMPTY"); return; }

        get_command_list().reset_auto_fence_count();
        if constexpr (false)
        {  
            constexpr bool get_kernel_time = false;

            bool prev_is_gpu;
            for (uint i = 0; i < sorted_nodes.size() ; i++)
            {
                auto tid = sorted_nodes[i];
                auto& task = list_task[tid];
                bool find; const auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                if (!find) 
                { 
                    if (prev_is_gpu)
                    {
                        std::vector<double> prev_costs_from_cmd_buffer = get_command_list().wait_all_cmd_buffers_and_get_costs(get_kernel_time);
                    }
                    {
                        imp.launch_task(task_to_param(task)); // task.print_with_cluster(tid);
                    }
                    prev_is_gpu = false; 
                }  
                else 
                {
                    auto buffer = get_command_list().start_new_list_with_new_buffer(); 
                    {
                        imp.launch_task(task_to_param(task));
                    }
                    get_command_list().make_fence_with_previous_cmd_buffer(); // If False, The Function May Be Empty
                    get_command_list().send_last_cmd_buffer_in_list();
                    prev_is_gpu = true;
                }
            }

            auto costs = get_command_list().wait_all_cmd_buffers_and_get_costs(get_kernel_time); 
            
        }

        for (uint i = 0; i < sorted_nodes.size(); i++) 
        {
            auto tid = sorted_nodes[i]; auto& task = list_task[tid];
            bool find; auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find); if(!find) { get_command_list().send_and_wait(); } 
            imp.launch_task(task_to_param(task));
        } 
        get_command_list().send_and_wait();

        // const uint sync_count = 40;
        // for (uint i = 0; i < list_order.size(); i++) 
        // {
        //     if ((i % sync_count == 0) && (i != list_order.size() - 1))
        //     {
        //         get_command_list().start_new_list_with_new_buffer();
        //     }
        //     auto tid = list_order[i]; auto& task = list_task[tid];
        //     bool find; auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find); if (!find) { get_command_list().send_and_wait(); } 
        //     imp.launch_task(task_to_param(task));
        //     if (i % sync_count == (sync_count - 1) || i == list_order.size() - 1)
        //     {
        //         get_command_list().make_fence_with_previous_cmd_buffer();
        //     }
        // } 
        // get_command_list().send_all_cmd_buffers();
        // get_command_list().wait_all_cmd_buffers();
        // get_command_list().clear_all_cmd_buffers();

    }
    else if (mode == LaunchModeCpu)
    {
        if (sorted_nodes.empty()) 
        { 
            fast_print_err("List Order Is EMPTY, Switch To Topology SortingOrder");
            for (uint i = 0; i < list_order.size(); i++) 
            {
                auto tid = list_order[i]; auto& task = list_task[tid];
                bool find; auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if(!find) { fast_print_err("There Is NO CPU Implementation!!"); return; }
                imp.launch_task(task_to_param(task));
            }  
        }
        else 
        {
            for (uint i = 0; i < sorted_nodes.size(); i++) 
            {
                auto tid = sorted_nodes[i]; auto& task = list_task[tid];
                bool find; auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if(!find) { fast_print_err("There Is NO CPU Implementation!!"); return; }
                imp.launch_task(task_to_param(task));
            } 
        }

        
    }
    ///
    /// <Heterogeneous Computing in Cloth Simulation>
    ///
    else if (mode == LaunchModeHetero) {
        
        if (launch_events.empty() || (launch_events[0].empty() && launch_events[1].empty())) { fast_print_err("Launching Order Is EMPTY"); return; }

        const bool print_schedule_event = false;
        
        /// Reset cmd
        
        get_shared_event().refresh();
        get_command_list().reset_auto_fence_count();
  
        /// Launch GPU Commands : Async
        const ListSchedule& gpu_schedules = proc_schedules[1];
        const std::vector<LaunchEvent>& gpu_event = launch_events[1];
        const uint num_gpu_events = gpu_event.size();

        std::vector< MTL::CommandBuffer* > list_cmd_buffer(num_gpu_events);
        auto& auto_fence_count = get_command_list().auto_fence_count;

        for (uint cmd_idx = 0; cmd_idx < num_gpu_events; cmd_idx++) 
        {
            
            const LaunchEvent& event = gpu_event[cmd_idx];

            // get_command_list().start_new_list();
            list_cmd_buffer[cmd_idx] = get_command_list().start_new_list_with_new_buffer();

            /// Wait & Signal Between CPU & CPU
            if (!fully_not_wait) 
            {
                if ( event.wait != -1u) 
                {
                    // fast_print(cmd_idx, "GPU Waits for CPU", event.wait + 1);
                    get_command_list().wait_cpu(get_shared_event(), event.wait);  
                    // assemble_impl[1]();
                }
            }
            
            /// Launch
            for (uint i = event.start_idx; i <= event.end_idx; i++) 
            {

                SimClock clock2; clock2.start_clock();

                auto& gpu_jobs = gpu_schedules[i];
                uint tid = gpu_jobs.task_id;
                auto& task = list_task[tid];

                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                if (find) 
                {
                    imp.launch_task(task_to_param(task)); // task.print_with_cluster(tid);
                }   
                else
                {
                    fast_print_err("Does NOT Have GPU Implementation, Wrong Dispatch Logic", taskNames.at(task.func_id));
                    return;
                }

            }   
            get_command_list().make_fence_with_previous_cmd_buffer(); // If False, The Function May Be Empty
            get_command_list().send_last_cmd_buffer_in_list();
        }

        /*
        for (uint cmd_idx = 0; cmd_idx < num_gpu_events; cmd_idx++) {
            
            const LaunchEvent& event = gpu_event[cmd_idx];

            {
                get_command_list().start_new_list();
            }

            /// Wait & Signal Between CPU & CPU
            if (!fully_not_wait) {
                if( event.wait != -1u) {
                    // fast_print(cmd_idx, "GPU Waits for CPU", event.wait + 1);
                    get_command_list().wait_cpu(get_shared_event(), event.wait);  
                }
            }
            
            /// Launch
            for (uint i = event.start_idx; i <= event.end_idx; i++) {

                auto& gpu_jobs = gpu_schedules[i];
                uint tid = gpu_jobs.task_id;
                auto& task = list_task[tid];

                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);

                if(find){
                    imp.launch_task(task_to_param(task));
                }   
                else{
                    fast_print_err("Does NOT Have GPU Implementation, Wrong Dispatch Logic", taskNames.at(task.func_id));
                    return;
                }
            }   

            list_cmd_buffer[cmd_idx] = get_command_list().command_buffer;

            if (auto_fence_count != 0) 
                get_command_list().wait_fence(get_fence(auto_fence_count++)); 
            auto_fence_count++; if (auto_fence_count == get_num_fence() - 1) { auto_fence_count = 0; }
            get_command_list().make_fence(get_fence(auto_fence_count)); 

            get_command_list().is_command_buffer_active = false; 
            get_command_list().current_task_num = 0;
            
        }

        for (auto& buffer : list_cmd_buffer) {
            buffer->commit();
        }
        */

        // Encode/Start/Get/Segment =  0.380/ 0.032/ 0.054/ 0.036
        // fast_format("Encode/Start/Get/Segment = {:6.3f}/{:6.3f}/{:6.3f}/{:6.3f}",total_encode, total_start, total_get, total_segment);

        ///
        /// Launch CPU Commands : Immediate
        ///
        const ListSchedule& cpu_schedules = proc_schedules[0];
        const std::vector<LaunchEvent>& cpu_event = launch_events[0];
        const uint num_cpu_events = cpu_event.size();
        
        std::vector<float> runtime_cost_cpu(num_cpu_events, 0.0f);
        for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) {
            const LaunchEvent& event = cpu_event[cmd_idx];

            // SimClock::sleep_ms(50);
            
            /// Wait
            if (!fully_not_wait){
                if(event.wait != -1u){
                    // SimClock clock; clock.start_clock();
                    list_cmd_buffer[event.wait]->waitUntilCompleted();
                    // float dt = clock.end_clock();
                }
            }
            

            /// Launch
            SimClock clock; clock.start_clock(); 
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& cpu_jobs = cpu_schedules[i];
                uint tid = cpu_jobs.task_id;
                auto& task = list_task[tid];

                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if (find) {
                    imp.launch_task(task_to_param(task));
                }   
                else {
                    fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                    return;
                }
            }   
            runtime_cost_cpu[cmd_idx] = (clock.end_clock());            

            if (event.signal != -1u) {
                get_shared_event().event->setSignaledValue(event.signal);
            }
        }


        // Refit CPU
        for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++)
        {
            const LaunchEvent& event = cpu_event[cmd_idx];
            auto dt = runtime_cost_cpu[cmd_idx];

            float profile_sum = 0.0f; uint un_fixed_count = 0;
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& cpu_jobs = cpu_schedules[i];
                uint tid = cpu_jobs.task_id;
                if (!list_task[tid].is_computation_time_constant) un_fixed_count++;
                profile_sum += computation_matrix[tid][0];
                if (i != event.end_idx) profile_sum += communication_cost_matrix_uma[0][0];
            } 
            float delta_sum = dt - profile_sum;
            float average_delta = delta_sum / float(un_fixed_count);
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& cpu_jobs = cpu_schedules[i];
                uint tid = cpu_jobs.task_id;
                if (!list_task[tid].is_computation_time_constant)
                    computation_matrix[tid][0] = 0.5f * max_scalar(computation_matrix[tid][0] + computation_matrix[tid][0] + average_delta, 0.0f); // Get Average
            } 
            
            // if (dt / profile_sum > 1.3f)
            // {
            //     fast_format("  CPU Cmd Idx {} : ProfileTime = {:6.3f}, Actually Get {:6.3f}", cmd_idx, profile_sum, clock.duration());
            //     for (uint i = event.start_idx; i <= event.end_idx; i++) {
            //         auto& cpu_jobs = cpu_schedules[i];
            //         uint tid = cpu_jobs.task_id;
            //         auto& task = list_task[tid];
            //         fast_format("     Out Of Range : task {} {}", tid, taskNames.at(task.func_id));
            //     } 
            // }
        }
        
        // Refit GPU
        
        for (uint cmd_idx = 0; cmd_idx < num_gpu_events; cmd_idx++) {
            const LaunchEvent& event = gpu_event[cmd_idx];

            float profile_sum = 0.0f; uint un_fixed_count = 0;
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& gpu_jobs = gpu_schedules[i];
                uint tid = gpu_jobs.task_id;
                if (!list_task[tid].is_computation_time_constant) un_fixed_count++;
                profile_sum += computation_matrix[tid][1];
                if (i != event.end_idx) profile_sum += communication_cost_matrix_uma[1][1];
            }   

            auto buffer = list_cmd_buffer[cmd_idx];
            float dt = (buffer->GPUEndTime() - buffer->GPUStartTime()) * 1000.0;

            float delta_sum = dt - profile_sum;
            float average_delta = delta_sum / float(un_fixed_count);
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& gpu_jobs = gpu_schedules[i];
                uint tid = gpu_jobs.task_id;
                if (!list_task[tid].is_computation_time_constant)
                    computation_matrix[tid][1] = 0.5f * max_scalar(computation_matrix[tid][1] + computation_matrix[tid][1] + average_delta, 0.0f);
            } 

            // if (cost / profile_sum > 1.3f)
            // {
            //     fast_format("  GPU Cmd Idx {} : ProfileTime = {:6.3f}, Actually Get {:6.3f}", cmd_idx, profile_sum, cost);
            //     for (uint i = event.start_idx; i <= event.end_idx; i++) {
            //         auto& gpu_jobs = gpu_schedules[i];
            //         uint tid = gpu_jobs.task_id;
            //         auto& task = list_task[tid];
            //         fast_format("     Out Of Range : task {} {}", tid, taskNames.at(task.func_id));
            //     } 
            // }
        }

    }
    else if (mode == LaunchModeProgressiveHetero) {
        
        if (launch_events.empty() || (launch_events[0].empty() && launch_events[1].empty())) { fast_print_err("Launching Order Is EMPTY"); return; }

        
        /// Launch GPU Commands : Async
        const ListSchedule& gpu_schedules = proc_schedules[1];
        const std::vector<LaunchEvent>& gpu_event = launch_events[1];
        const uint num_gpu_events = gpu_event.size();

        // std::vector< MTL::CommandBuffer* > list_cmd_buffer(num_gpu_events);
        std::vector< MTL::CommandBuffer* > list_cmd_buffer; list_cmd_buffer.reserve(num_gpu_events);
        auto& auto_fence_count = get_command_list().auto_fence_count;

        /// Reset cmd
        // for (uint i = 0; i < list_cmd_buffer.size(); i++) {
        //     get_shared_event(i).event->setSignaledValue(0);
        //     get_shared_event(i).refresh();
        // }

        get_shared_event().refresh();
        get_command_list().reset_auto_fence_count();

        const uint max_cmd_buffer_count = 15; const uint reserve_cmd_count = 0;
        uint merged_event_idx = 0;

        auto fn_launch_merged_gpu_events = [&]()
        {
            const uint bg = max_cmd_buffer_count * merged_event_idx;
            const uint ed = min_scalar(max_cmd_buffer_count * (merged_event_idx + 1), num_gpu_events);
            merged_event_idx++;

            if (bg >= num_gpu_events) return ;

            for (uint cmd_idx = bg; cmd_idx < ed; cmd_idx++) 
            {
                const LaunchEvent& event = gpu_event[cmd_idx];

                // get_command_list().start_new_list();
                // list_cmd_buffer[cmd_idx] = get_command_list().start_new_list_with_new_buffer();
                list_cmd_buffer.push_back(get_command_list().start_new_list_with_new_buffer()); // Insert Into cmd_idx 
            
                if (!fully_not_wait) 
                {
                    if ( event.wait != -1u) 
                    {
                        // fast_print(cmd_idx, "GPU Waits for CPU", event.wait + 1);
                        get_command_list().wait_cpu(get_shared_event(), event.wait);  
                        // assemble_impl[1]();
                    }
                }
                
                /// Launch
                for (uint i = event.start_idx; i <= event.end_idx; i++) 
                {

                    SimClock clock2; clock2.start_clock();

                    auto& gpu_jobs = gpu_schedules[i];
                    uint tid = gpu_jobs.task_id;
                    auto& task = list_task[tid];

                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                    if (find) 
                    {
                        imp.launch_task(task_to_param(task)); // task.print_with_cluster(tid);
                    }   
                    else
                    {
                        fast_print_err("Does NOT Have GPU Implementation, Wrong Dispatch Logic", taskNames.at(task.func_id));
                        return;
                    }
                }   

                if (cmd_idx != bg) get_command_list().make_fence_with_previous_cmd_buffer(); // If False, The Function May Be Empty
                get_command_list().send_last_cmd_buffer_in_list();
                // get_command_list().is_command_buffer_active = false;
                // get_command_list().current_task_num = 0;
            }
        };

          
        fn_launch_merged_gpu_events();

    
        // fast_print("Proc 1 : ");

        // for (auto& buffer : list_cmd_buffer) {
        //     buffer->commit();
        // }

        ///
        /// Launch CPU Commands : Immediate
        ///
        const ListSchedule& cpu_schedules = proc_schedules[0];
        const std::vector<LaunchEvent>& cpu_event = launch_events[0];
        const uint num_cpu_events = cpu_event.size();
        
        // fast_print("Proc 0 : ");
        for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) 
        {
            const LaunchEvent& event = cpu_event[cmd_idx];

            /// Wait
            if (event.wait != -1u) 
            {
                if (event.wait >= list_cmd_buffer.size() - 1 - reserve_cmd_count)
                {
                    get_command_list().wait_all_cmd_buffers(); 
                    get_shared_event().refresh();
                    get_command_list().reset_auto_fence_count();
                    fast_format("Resend Cmd Buffer In CPU Cmd {} , Launch From {} to {}", cmd_idx, 
                        max_cmd_buffer_count * merged_event_idx, min_scalar(max_cmd_buffer_count * (merged_event_idx + 1), num_gpu_events));
                    fn_launch_merged_gpu_events();
                }
                // SimClock clock; clock.start_clock();
                list_cmd_buffer[event.wait]->waitUntilCompleted();
                list_cmd_buffer[event.wait] = nullptr;
                // float dt = clock.end_clock();
            }

            /// Launch
            for (uint i = event.start_idx; i <= event.end_idx; i++) 
            {
                auto& cpu_jobs = cpu_schedules[i];
                uint tid = cpu_jobs.task_id;
                auto& task = list_task[tid];
                
                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if (find) 
                {
                    imp.launch_task(task_to_param(task));
                }   
                else 
                {
                    fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                    return;
                }
            }   

            if (event.signal != -1u) 
            {
                get_shared_event().event->setSignaledValue(event.signal);
            }
        }

        get_command_list().wait_all_cmd_buffers();

        // list_cmd_buffer.back()->waitUntilCompleted();

        // for (uint i = 0; i < list_cmd_buffer.size(); i++) {
        //     get_shared_event(i).event->setSignaledValue(0);
        // }

    }
    else if (mode == LaunchModeFakeHetero) {

         const bool print_schedule_event = false;

        const ListSchedule& gpu_schedules = proc_schedules[1];
        const std::vector<LaunchEvent>& gpu_event = launch_events[1];
        const uint num_gpu_events = gpu_event.size();
        
        const ListSchedule& cpu_schedules = proc_schedules[0];
        const std::vector<LaunchEvent>& cpu_event = launch_events[0];
        const uint num_cpu_events = cpu_event.size();

        auto fn_launch_gpu = [&](uint begin_cmd_idx, uint end_cmd_idx){
            for (uint gpu_cmd_idx = begin_cmd_idx; gpu_cmd_idx <= end_cmd_idx; gpu_cmd_idx++){

                const LaunchEvent& event = gpu_event[gpu_cmd_idx];
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& gpu_jobs = gpu_schedules[i];
                    uint tid = gpu_jobs.task_id;
                    auto& task = list_task[tid];

                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                    if (find){
                        // imp.launch_task({task.func_id, 0, task.num_threads, 256, 0, 0, true, false, true});
                        imp.launch_task(task_to_param(task)); // task.print_with_cluster(tid);
                    }   
                    else{
                        fast_print_err("Does NOT Have GPU Implementation, Wrong Dispatch Logic");
                        return;
                    }
                }   
            }
        };
        auto fn_launch_gpu_by_cpu = [&](uint begin_cmd_idx, uint end_cmd_idx){
            for (uint gpu_cmd_idx = begin_cmd_idx; gpu_cmd_idx <= end_cmd_idx; gpu_cmd_idx++) {
                const LaunchEvent& event = gpu_event[gpu_cmd_idx];
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& gpu_jobs = gpu_schedules[i];
                    uint tid = gpu_jobs.task_id;
                    auto& task = list_task[tid];

                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                    if (find) {
                        imp.launch_task(task_to_param(task));
                    }   
                    else {
                        fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                        return;
                    }
                }   
            }
        };
        auto fn_launch_cpu = [&](uint begin_cmd_idx, uint end_cmd_idx){
            for (uint cpu_cmd_idx = begin_cmd_idx; cpu_cmd_idx <= end_cmd_idx; cpu_cmd_idx++) {
                const LaunchEvent& event = cpu_event[cpu_cmd_idx];
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& cpu_jobs = cpu_schedules[i];
                    uint tid = cpu_jobs.task_id;
                    auto& task = list_task[tid];

                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                    if (find) {
                        imp.launch_task(task_to_param(task));
                    }   
                    else {
                        fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                        return;
                    }
                }   
            }
        };

        const bool use_gpu_scene = false;

        uint gpu_new_begin = 0;
        if constexpr (use_gpu_scene)
        {
            for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) {
                const LaunchEvent& event = cpu_event[cmd_idx];
                if (event.wait != -1u) {
                    fn_launch_gpu(gpu_new_begin, event.wait);
                    gpu_new_begin = event.wait + 1;
                }
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& cpu_jobs = cpu_schedules[i];
                    uint tid = cpu_jobs.task_id;
                    auto& task = list_task[tid];
                    
                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                    if (find) {
                        imp.launch_task(task_to_param(task)); // task.print_with_cluster(tid);
                    }   
                    else {
                        get_command_list().send_and_wait();
                        imp.launch_task(task_to_param(task));
                    }
                }   
            }
            fn_launch_gpu(gpu_new_begin, launch_events[1].size() - 1);
            get_command_list().send_and_wait();
        }
        else 
        {
            // fn_launch_gpu(gpu_new_begin, gpu_event.size() - 1); uint gpu_new_begin = 0;
            for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) {
                const LaunchEvent& event = cpu_event[cmd_idx];
                if (event.wait != -1u) {
                    // fast_format("GPU Part");
                    fn_launch_gpu(gpu_new_begin, event.wait);
                    gpu_new_begin = event.wait + 1;
                }
                get_command_list().send_and_wait();
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& cpu_jobs = cpu_schedules[i];
                    uint tid = cpu_jobs.task_id;
                    auto& task = list_task[tid];
                    
                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                    if (find) {
                        imp.launch_task(task_to_param(task)); // task.print_with_cluster(tid);
                    }   
                    else {
                        fast_format_err("Empty CPU Implementation");
                    }
                }   
            }
            fn_launch_gpu(gpu_new_begin, launch_events[1].size() - 1);
            get_command_list().send_and_wait();
        }   

    }
    else if (mode == LaunchModeSequeceHetero) {

        const bool print_schedule_event = false;
        const bool print_task = false;
        
        const ListSchedule& gpu_schedules = proc_schedules[1];
        const std::vector<LaunchEvent>& gpu_event = launch_events[1];
        const uint num_gpu_events = gpu_event.size();
        
        const ListSchedule& cpu_schedules = proc_schedules[0];
        const std::vector<LaunchEvent>& cpu_event = launch_events[0];
        const uint num_cpu_events = cpu_event.size();

        auto fn_launch_gpu_by_cpu = [&](uint begin_cmd_idx, uint end_cmd_idx){
            for (uint gpu_cmd_idx = begin_cmd_idx; gpu_cmd_idx <= end_cmd_idx; gpu_cmd_idx++) {
                const LaunchEvent& event = gpu_event[gpu_cmd_idx];
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& gpu_jobs = gpu_schedules[i];
                    uint tid = gpu_jobs.task_id;
                    auto& task = list_task[tid]; 

                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find); if (print_task) task.print_with_cluster(tid);
                    if (find) {
                        imp.launch_task(task_to_param(task));
                    }   
                    else {
                        fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                        return;
                    }
                }   
            }
        };
        auto fn_launch_cpu = [&](uint begin_cmd_idx, uint end_cmd_idx){
            for (uint cpu_cmd_idx = begin_cmd_idx; cpu_cmd_idx <= end_cmd_idx; cpu_cmd_idx++) {
                const LaunchEvent& event = cpu_event[cpu_cmd_idx];
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& cpu_jobs = cpu_schedules[i];
                    uint tid = cpu_jobs.task_id;
                    auto& task = list_task[tid]; 

                    bool find;
                    auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find); if (print_task) task.print_with_cluster(tid);
                    if (find) {
                        imp.launch_task(task_to_param(task));
                    }   
                    else {
                        fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                        return;
                    }
                }   
            }
        };

        
        uint gpu_new_begin = 0; 
        for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) {
            const LaunchEvent& event = cpu_event[cmd_idx];
            
            /// Wait
            if (event.wait != -1u) {
                if (print_task) fast_format("====== GPU Begin ======");
                fn_launch_gpu_by_cpu(gpu_new_begin, event.wait);
                gpu_new_begin = event.wait + 1;

                if (print_task) fast_format("====== GPU End ======\n");
            }

            if (print_task) fast_format("====== CPU Begin ======");
            /// Launch
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& cpu_jobs = cpu_schedules[i];
                uint tid = cpu_jobs.task_id;
                auto& task = list_task[tid];

                // fast_print("cpu", Launcher::taskNames.at(task.func_id));
                
                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find); if (print_task) task.print_with_cluster(tid);
                if (find) {
                    imp.launch_task(task_to_param(task)); 
                }   
                else {
                    fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                    return;
                }
            }   
            if (print_task) fast_format("====== CPU End ======\n");
        }
        fn_launch_gpu_by_cpu(gpu_new_begin, launch_events[1].size() - 1);
    }
    else if (mode == LaunchModePartialCPU) {
        ///
        /// Launch CPU Commands : Immediate
        ///
        const ListSchedule& cpu_schedules = proc_schedules[0];
        const std::vector<LaunchEvent>& cpu_event = launch_events[0];
        const uint num_cpu_events = cpu_event.size();

        for (uint cmd_idx = 0; cmd_idx < num_cpu_events; cmd_idx++) {
            const LaunchEvent& event = cpu_event[cmd_idx];
            
            SimClock clock; clock.start_clock(); float profile_sum = 0;

            /// Launch
            for (uint i = event.start_idx; i <= event.end_idx; i++) {
                auto& cpu_jobs = cpu_schedules[i];
                uint tid = cpu_jobs.task_id;
                auto& task = list_task[tid];
                
                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if(find){
                    imp.launch_task(task_to_param(task)); profile_sum += computation_matrix[tid][0];
                }   
                else{
                    fast_print_err("Does NOT Have CPU Implementation, Wrong Dispatch Logic");
                    return;
                }
            }   
            if (clock.duration() / profile_sum > 1.3f)
            {
                fast_format("  Cmd Idx {} : ProfileTime = {:6.3f}, Actually Get {:6.3f}", cmd_idx, profile_sum, clock.end_clock());
                for (uint i = event.start_idx; i <= event.end_idx; i++) {
                    auto& cpu_jobs = cpu_schedules[i];
                    uint tid = cpu_jobs.task_id;
                    auto& task = list_task[tid];
                    fast_format("     Out Of Range : task {} {}", tid, taskNames.at(task.func_id));
                } 
            }
        }

    }
    else if (mode == LaunchModePartialGPU) {
  
        /// Launch GPU Commands : Async
        const ListSchedule& gpu_schedules = proc_schedules[1];
        const std::vector<LaunchEvent>& gpu_event = launch_events[1];
        const uint num_gpu_events = gpu_event.size();


        for (uint cmd_idx = 0; cmd_idx < num_gpu_events; cmd_idx++) {
            const LaunchEvent& event = gpu_event[cmd_idx];

            get_command_list().start_new_list_with_new_buffer();
            // get_command_list().start_new_list();

            for (uint i = event.start_idx; i <= event.end_idx; i++) 
            {
                auto& gpu_jobs = gpu_schedules[i];
                uint tid = gpu_jobs.task_id;
                auto& task = list_task[tid];

                bool find;
                auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
                if(find)
                {
                    imp.launch_task(task_to_param(task));
                }   
                else
                {
                    fast_print_err("Does NOT Have GPU Implementation, Wrong Dispatch Logic", taskNames.at(task.func_id));
                    return;
                }
            }   

            get_command_list().make_fence_with_previous_cmd_buffer();
            get_command_list().send_last_cmd_buffer_in_list();
            // if(cmd_idx != 0)                    { get_command_list().wait_fence(get_fence(cmd_idx - 1)); }
            // if(cmd_idx != num_gpu_events - 1)   { get_command_list().make_fence(get_fence(cmd_idx));  } 
            // get_command_list().send_list();   
        }
        get_command_list().wait_all_cmd_buffers();

        // for (uint i = 0; i < proc_schedules[1].size(); i++)
        // {   
        //     const auto& task_info = proc_schedules[1][i];
        //     const auto& task = list_task[task_info.task_id];
        //     bool find; auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
        //     get_command_list().start_new_list();
        //     {
        //         imp.launch_task(task_to_param(task));
        //     }
        //     // Not related to Fence
        //     // if(i != 0)                  { get_command_list().wait_fence(get_fence(i - 1)); }
        //     // if(i != num_gpu_events - 1) {  get_command_list().make_fence(get_fence(i)); } 
        //     get_command_list().send_list();
        // }
        
        // for (const auto& task_info : proc_schedules[1])
        // {   
        //     const auto& task = list_task[task_info.task_id];
        //     bool find; auto& imp = task.get_implementation(Launcher::DeviceTypeGpu, find);
        //     imp.launch_task(task_to_param(task));
        // }
        
    }

    #endif
   
}

/// Root node was not the first node in the sorted list. 
/// Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first
/// You need to provide a function that does nothing on each devices (for gpu, it is a empty kernel, for making commmand buffer not empty to be uploaded)
void Scheduler::standardizing_dag(const std::vector< std::function<void(const Launcher::LaunchParam&)> >& input_list_fn_empty_func) {

    ///
    /// Set root , if more than one root, set the fastest one
    /// Nodes with no successors cause the any expression to be empty
    /// 
    // fast_print("root node", Launcher::taskNames.at(list_task[root_node].func_id));
    // fast_print("terminal node", Launcher::taskNames.at(list_task[terminal_node].func_id));

    std::vector<uint> list_root;
    std::vector<uint> list_terminal;

    // auto fn_empty_func = [](const Launcher::LaunchParam&) -> void{ };

    for (uint tid = 0; tid < list_task.size(); tid++) {
        const auto& task = list_task[tid];
        if (task.predecessors.empty()){ // 前置节点为空
            list_root.push_back(tid);
        }
    }
    const bool print_root_terminal = false;

    const uint num_procs = communication_startup.size();
    std::vector<Implementation> list_empty_imp = {};
    if (!input_list_fn_empty_func.empty()) {
        for (uint proc = 0; proc < num_procs; proc++) {
            list_empty_imp.push_back(Implementation(proc, input_list_fn_empty_func[proc]));
        }
    }
    else {
        for (uint proc = 0; proc < num_procs; proc++) {
            list_empty_imp.push_back(Implementation(
                // proc, [proc](const Launcher::LaunchParam& param){ fast_format("I am additional root or terminal node in proc {}", proc); }
                proc, [](const Launcher::LaunchParam& param){}
            ));
        }
    }
    
    // if(list_root.size() > 1) 
    if (true) {
        root_node = add_task(Task(id_additional_root, 0, false, true, 
            list_empty_imp));
            // { Implementation(DeviceTypeCpu, fn_empty_func, 0), Implementation(DeviceTypeGpu, fn_empty_func, 0) }));

        computation_matrix.push_back(ListCost(num_procs, 0.0f));
        list_order.insert(list_order.begin(), root_node);

        if(print_root_terminal) fast_print("   Expected a Single Root Node, Switch to", 
                    Launcher::taskNames.at(list_task[root_node].func_id));
                    // Launcher::taskNames.at(list_task[root_node].func_id), "Change the Following Nodes Linking to the Root");
        
        // fast_print(" Node Connect To Root Node is :");
        for (auto tid : list_root) {
            if (tid != root_node) {
                // std::cout << "   " << tid << " : " << Launcher::taskNames.at(list_task[tid].func_id) << std::endl;
                set_connect(root_node, tid, 0.0f);
            }
        }
    }
    else {
        root_node = list_root[0];
        if (print_root_terminal) fast_print("   Root Node", Launcher::taskNames.at(list_task[root_node].func_id));
        if (!list_task[root_node].predecessors.empty()){
            fast_print_err(" ======= Root Node HAS Predecessors ======= ");
        }
    }

    /// Normalize根节点的操作可能会更改依赖情况
    for (uint tid = 0; tid < list_task.size(); tid++) {
        const auto& task = list_task[tid];
        if(task.successors.empty()){
            list_terminal.push_back(tid);
        }
    }

    // if(list_terminal.size() > 1) 
    if (true) {
        terminal_node = add_task(Task(id_additional_terminal, 0, false, true, 
            list_empty_imp));
            // { Implementation(DeviceTypeCpu, fn_empty_func, 0), Implementation(DeviceTypeGpu, fn_empty_func, 0) }));
        computation_matrix.push_back(ListCost(num_procs, 0.0f));
        list_order.push_back(terminal_node);

        
        if (print_root_terminal) fast_print("   Expected a Terminal Root Node, Switch to", Launcher::taskNames.at(list_task[terminal_node].func_id));
        // if(print_root_terminal) fast_print("   Expected a Terminal Root Node, Switch to", Launcher::taskNames.at(list_task[terminal_node].func_id), "Change the Following Nodes Linking to the Terminal");

        // fast_print(" Node Connect To Terminal Node is");
        for (auto tid : list_terminal){
            if(tid != terminal_node){
                // std::cout << "   " << tid << " : " << Launcher::taskNames.at(list_task[tid].func_id) << std::endl;
                set_connect(tid, terminal_node, 0.f);
            }
        }
        
    }
    else {
        terminal_node = list_terminal[0];
        if (print_root_terminal) fast_print("   Terminal Node", Launcher::taskNames.at(list_task[terminal_node].func_id));
    }


    const bool print_connectivity_matrix = false; /// --- Print --- 
    if (print_connectivity_matrix){
        const uint num_tasks = list_task.size();
        
        // T  ,T0 ,T1 ,T2 ,T3 ,T4 ,T5 ,T6 ,T7 ,T8 ,T9
        std::vector< std::vector<float> > list_connectivity(num_tasks);
        for (uint tid = 0; tid < num_tasks; tid++){
            auto& rows = list_connectivity[tid];
            rows.resize(num_tasks, 0.f);
            const auto& task = list_task[tid];
            for (uint j = 0; j < task.successors.size(); j++) {
                const uint front = task.successors[j];
                const uint weight = task.list_weight[j];
                rows[front] = weight;
            }
        }
        std::cout << "list_connectivity : \n" << "T";
        for (uint tid = 0; tid < num_tasks; tid++){
            std::cout << ", T" << tid;
        }
        std::cout << "\n";
        for (uint i = 0; i < list_task.size(); i++) {
            std::cout << "T" << i ;
            for (uint j = 0; j < list_task.size(); j++){
                std::cout << ", " << list_connectivity[i][j];
            }
            fast_print("");
        }
    }
}

#define USE_MERGED_TASK false

void Scheduler::compute_ranku(uint num_procs)
{

    const uint num_tasks = list_task.size();
    if (num_tasks == 0) 
    {
        fast_print_err("Task List is Empty");
        return;
    }
    // for(uint tid = 0; tid < num_tasks; tid++){
    //     auto& task = list_task[tid];
    //     std::cout << "   " << tid << "(" << taskNames.at(task.func_id) << ") : (pred) "; for (auto& pred : task.predecessors) { fast_print_single(pred); }
    //     std::cout << " (succ) ";  for (auto& succ : task.successors) { fast_print_single(succ); } std::cout << "\n";
    // }


    ///
    /// Calculate average communication cost
    /// Should be communication speed ? Cause the larger the speed, the less time use
    /// avgCommunicationCost = 1.0 in python_heft
    ///
    float avg_comm_startup = 0.f; // \hat{ L } is average latency time of all preocessors // 延迟
    float sum_communication_cost = 0.0; float sum_bandwidth = 0.0;
    for (uint proc1 = 0; proc1 < num_procs; proc1++) 
    {
        avg_comm_startup += communication_startup[proc1];
        for (uint proc2 = 0; proc2 < num_procs; proc2++) 
        {
            if (proc1 != proc2) 
            {
                if (!communication_cost_matrix_uma.empty()) sum_communication_cost += communication_cost_matrix_uma[proc1][proc2];
                if (!communication_speed_matrix.empty()) sum_bandwidth += communication_speed_matrix[proc1][proc2];
            }
        }
    }

    const float avgCommunicationCost = 1.0 + avg_comm_startup; // speed
    // const float avgCommunicationCost = sum_communication_cost / float(num_procs * (num_procs - 1));
    const float avgCommunicationBandWidth = sum_bandwidth / float(num_procs * (num_procs - 1));
    std::vector< std::vector<float> > list_avg_weight(num_tasks);
    for (uint tid = 0; tid < num_tasks; tid++) 
    {
        // \hat{ c_ij } = \hat{ L } + data_ij / \hat{ B }
        //                                      \hat { B } is average bandwidth of all links connecting the set of P processors
        auto& task = list_task[tid];

        auto& weights = task.list_weight; // num
        auto& avg_weights = list_avg_weight[tid];
        avg_weights.resize(weights.size());

        for (uint j = 0; j < weights.size(); j++) 
        {
            // avg_weights[i] = avg_comm_startup + weights[i] / avgCommunicationBandWidth; // comm_time = length(data) / bandwidth
            avg_weights[j] = avg_comm_startup + avgCommunicationCost;
        }
    }

    ///
    /// Utilize a masked array so that np.mean, etc, calculations ignore the entries that are inf
    ///
    std::vector< ListMask > comp_matrix_masked(num_tasks, {true, true, true});
    for (uint tid = 0; tid < num_tasks; tid++) 
    {
        auto& costs = computation_matrix[tid];
        auto& mask = comp_matrix_masked[tid];
        for (uint proc = 0; proc < num_procs; proc++) 
        {
            mask[proc] = costs[proc] != inf;
        }
    }

    std::vector<float> list_avg_cost(num_tasks, 0.0f);
    auto fn_compute_avg_cost = [&](uint tid) -> double { 
        const ListMask& comp_msk = comp_matrix_masked[tid];
        const ListCost& cost = computation_matrix[tid];
        double avg_cost = 0.0, num = 0.0;
        for (uint proc = 0; proc < num_procs; proc++) {
            if(comp_msk[proc]) { 
                avg_cost += cost[proc]; 
                num += 1.0; 
            }
        }
        avg_cost /= num;
        return avg_cost;
    };
    auto fn_compute_max_cost = [&](uint tid) -> std::pair<uint, float> { 
        const ListMask& comp_msk = comp_matrix_masked[tid];
        const ListCost& costs = computation_matrix[tid];

        uint max_cost_proc = 0;
        float max_cost = -1.0f;
        
        for (uint proc = 0; proc < num_procs; proc++) {
            if(comp_msk[proc]){
                if (costs[proc] > max_cost) {
                    max_cost_proc = proc;
                    max_cost = costs[proc];
                }
            }
        }
        return std::make_pair(max_cost_proc, max_cost);
    };
    auto fn_compute_min_cost = [&](uint tid) -> std::pair<uint, float> { 
        const ListMask& comp_msk = comp_matrix_masked[tid];
        const ListCost& costs = computation_matrix[tid];

        uint min_cost_proc = 0;
        float min_cost = inf;
        
        for (uint proc = 0; proc < num_procs; proc++) {
            if(comp_msk[proc]){
                if (costs[proc] < min_cost) {
                    min_cost_proc = proc;
                    min_cost = costs[proc];
                }
            }
        }
        return std::make_pair(min_cost_proc, min_cost);
    };
    for (uint tid = 0; tid < num_tasks; tid++) {
        list_avg_cost[tid] = fn_compute_avg_cost(tid);
    }
    

    //
    // Compute RankU From Right To Left
    //
    const RankMetric metric = RankMetric::RankMetricMEAN; //  RankMetricMEAN // RankMetricOCT
    // const RankMetric metric = RankMetric::RankMetricWORST; // RankMetricWORST // RankMetricBEST（Failed) // RankMetricEDP (Requeres Energy Dictionary)

#if USE_MERGED_TASK

{
    //
    // Prev Processing : Merge Small Tasks 
    //
    std::vector<float> list_avg_costs_merged;
    list_task_merged.clear(); task_to_merged_task_map.resize(num_tasks); 
    const auto list_order_dfs = topological_sort_DFS(list_task); 
    // const float threshold_time = 0.5 * avgCommunicationCost;
    // const float threshold_time = 0.8 * avgCommunicationCost;
    const float threshold_time = avgCommunicationCost;
    const float average_communication_matrix_inner_device = (communication_cost_matrix_uma[0][0] + communication_cost_matrix_uma[1][1]) / num_procs;

    // Merge Into A Larger One
    {
        auto current_type = list_task[list_order_dfs[0]].constraint_idx;
        double current_time_sum = 0.0; std::vector<uint> curr_cluster; 
        bool prev_has_only_one_implementation = false;

        auto fn_insert_merged_task = [&]() {
            const uint merged_tid = list_task_merged.size();
            list_task_merged.push_back(MergedTask(curr_cluster, current_type));
            list_avg_costs_merged.push_back(current_time_sum);
            for (const uint tid : curr_cluster) {
                task_to_merged_task_map[tid] = merged_tid;
            }
            // fast_format("  Cluster {:2} : ", merged_tid);
            // for (uint tid : curr_cluster) { const auto& task = list_task[tid]; fast_format("    {}", taskNames.at(task.func_id)); }
        };

        for (const uint tid : list_order_dfs) 
        {
            const auto& task = list_task[tid];
            const auto curr_cost = list_avg_cost[tid];
            const bool curr_has_only_one_implementation = (task.list_implementation.size() < 2);
            const bool need_to_segment = (task.constraint_idx != current_type) 
                                      || (current_time_sum + curr_cost + average_communication_matrix_inner_device >= threshold_time) 
                                      || curr_has_only_one_implementation || prev_has_only_one_implementation
                                      ; 
            if (need_to_segment)
            { 
                if (!curr_cluster.empty()) { fn_insert_merged_task(); }
                curr_cluster = {tid};
                current_type = task.constraint_idx;
                current_time_sum = curr_cost;
            } 
            else 
            {
                curr_cluster.push_back(tid);
                current_time_sum += curr_cost + average_communication_matrix_inner_device;
            }
            prev_has_only_one_implementation = curr_has_only_one_implementation;
        } if (!curr_cluster.empty()) { fn_insert_merged_task(); }
    }

    const uint num_tasks_merged = list_task_merged.size(); 
    uint sum = 0; for (const auto& list : list_task_merged) { sum += list.tasks.size();  }  
    fast_format("NumTasks Clusterd = {} (Sum of InnerTask = {}, Desire For {})", num_tasks_merged, sum, num_tasks);

    
    // Fill-in Connections
    {
        for (uint merged_tid = 0; merged_tid < list_task_merged.size(); merged_tid++) 
        {
            auto& merged_task = list_task_merged[merged_tid];

            std::set<uint> set_predecessors;
            std::set<uint> set_successors;            

            std::map<uint, float> weight_map;

            for (uint j = 0; j < merged_task.tasks.size(); j++) 
            {
                const uint tid = merged_task.tasks[j];
                const auto& task = list_task[tid];

                for (const uint pred : task.predecessors) 
                { 
                    const uint merged_tid_pred = task_to_merged_task_map[pred]; 
                    if (merged_tid_pred != merged_tid) { set_predecessors.insert(merged_tid_pred); } 
                }
                
                for (uint jj = 0; jj < task.successors.size(); jj++)
                { 
                    const uint succ = task.successors[jj];
                    const uint merged_tid_succ = task_to_merged_task_map[succ]; 
                    if (merged_tid_succ != merged_tid) 
                    {  
                        set_successors.insert(merged_tid_succ);  
                        const float curr_weight = task.list_weight[jj];
                        if (!weight_map.contains(merged_tid_succ)) // Simple Addition : Byte of Data Transfer Between Merged Data Is Smaller Than Their Summary
                            weight_map[merged_tid_succ] = curr_weight;
                        else 
                            weight_map[merged_tid_succ] += curr_weight;
                    }
                }
            }
            merged_task.predecessors.assign(set_predecessors.begin(), set_predecessors.end());
            merged_task.successors.assign(set_successors.begin(), set_successors.end());
            for (const uint succ : set_successors) { merged_task.weights.push_back(weight_map[succ]); }

        } 
    } 

    // Check
    {
        for (uint merged_tid = 0; merged_tid < list_task_merged.size(); merged_tid++) 
        {
            auto& merged_task = list_task_merged[merged_tid];
            
            // Check Constraint Index Is Same & Check If Task With Single Implementation Task Is The Unique Task In Merged Task
            auto constraint_idx = merged_task.constraint_idx;
            for (const uint tid : merged_task.tasks) 
            {
                const auto& task = list_task[tid];
                if (task.constraint_idx != constraint_idx) 
                {
                    fast_print_err("Constraint Index In Single Merged Task Is NOT Same");
                }
                if (task.list_implementation.size() < 2 && merged_task.tasks.size() != 1) 
                {
                    fast_print_err("Task With Single Implementation Task Is NOT The Unique Task In Merged Task");
                }
            }

            // Check Connection Inner Merged Cluster
            for (uint j = 0; j < merged_task.tasks.size(); j++) 
            {
                const uint tid = merged_task.tasks[j];
                const auto& task = list_task[tid];
                for (auto pred : task.predecessors) 
                { 
                    const uint pred_belongs = task_to_merged_task_map[pred];
                    if (pred_belongs == merged_tid) 
                    { 
                        uint jj = std::find(merged_task.tasks.begin(), merged_task.tasks.end(), pred) - merged_task.tasks.begin();
                        if (jj >= j) { fast_print_err("Miss Connection Inner Merged Task!!!"); }
                    } 
                    else 
                    {
                        if (pred_belongs > merged_tid) { fast_print_err("Miss Connection Outer Merged Task!!!"); }
                    }
                }
                for (auto succ : task.successors) 
                { 
                    const uint succ_belongs = task_to_merged_task_map[succ];
                    if (task_to_merged_task_map[succ] == merged_tid) 
                    {
                        uint jj = std::find(merged_task.tasks.begin(), merged_task.tasks.end(), succ) - merged_task.tasks.begin();
                        if (jj <= j) { fast_print_err("Miss Connection Inner Merged Task!!!"); }
                    } 
                    else 
                    {
                        if (succ_belongs < merged_tid) { fast_print_err("Miss Connection Outer Merged Task!!!"); }
                    }
                }
            }

            // Check Connection Between Merged Clusters
            for (auto pred : merged_task.predecessors) 
            { 
                if (pred >= merged_tid) fast_print_err("Index of Pred Node Should Not Larger / Equal Than Self");
                if (std::find(list_task_merged[pred].successors.begin(), list_task_merged[pred].successors.end(), merged_tid) == list_task_merged[pred].successors.end()) 
                    fast_print_err("Connection Not Correspond!!"); 
            }
            for (auto succ : merged_task.successors)   
            { 
                if (succ <= merged_tid) fast_print_err("Index of Succ Node Should Not Small / Equal Than Self");
                if (std::find(list_task_merged[succ].predecessors.begin(), list_task_merged[succ].predecessors.end(), merged_tid) == list_task_merged[succ].predecessors.end()) 
                    fast_print_err("Connection Not Correspond!!");
            }
            
            // fast_format("Merged Task {:2} : ", merged_tid); 
            // fast_print_single("            : Task = "); for (auto tid : merged_task.tasks) fast_print_single(taskNames.at(list_task[tid].func_id)); fast_print();
            // fast_print_single("            : Pred = "); for (auto pred : merged_task.predecessors) fast_print_single(pred); fast_print();
            // fast_print_single("            : Succ = "); for (auto succ : merged_task.successors) fast_print_single(succ);   fast_print();
        }
    } 

    // Fill-in Costs
    {
        for (auto& merged_task : list_task_merged) 
        {
            std::vector<float>& sum_costs = merged_task.costs;
            sum_costs.resize(num_procs, 0.0);
            for (const uint tid : merged_task.tasks) 
            {
                for (uint proc = 0; proc < num_procs; proc++) 
                {
                    // If 'computation_matrix[tid][proc] == inf', Then The Cost of Merged Task is inf... So We Need To Segment Tasks With Single Implement
                    sum_costs[proc] += computation_matrix[tid][proc];
                }
            }
            for (uint proc = 0; proc < num_procs; proc++) 
            {
                const auto comm_cost_inner_proc = communication_cost_matrix_uma[proc][proc];
                sum_costs[proc] += (merged_task.tasks.size() - 1) * comm_cost_inner_proc; // 1 -> 2 -> 3 : 2 Inner Connection
            }
        }
    }

    // fast_print_iterator(list_avg_cost, "list_avg_cost");
    // fast_print_iterator(list_avg_costs_merged, "list_avg_costs_merged");

    //
    // Compute RankU of Merged Task(Based On Merged Ralationship)
    //
    // std::vector<uint> list_order_merged = topological_sort_BFS(list_task_merged);
    ranku.resize(num_tasks_merged, 0.0);
    ranku.back() = list_avg_costs_merged.back();

    for (uint i = 1; i < num_tasks_merged; i++) 
    {
        // const uint merged_tid = list_order_merged[num_tasks_merged - 1 - i];
        const uint merged_tid = num_tasks_merged - 1 - i;
        const auto& merged_task = list_task_merged[merged_tid];    
        if (metric == RankMetric::RankMetricMEAN) 
        {
            double max_successor_ranku = -1.0f;
            for (const uint succ : merged_task.successors) 
            {
                double val = avgCommunicationCost + ranku[succ]; // avg_weight
                if (ranku[succ] == 0.0) fast_print_err("Access Zero RankU From Successors!!!!!"); 
                max_successor_ranku = max_scalar(max_successor_ranku, val);
            }
            if (max_successor_ranku < 0.f) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << max_successor_ranku << "}\n";
            ranku[merged_tid] = list_avg_costs_merged[merged_tid] + max_successor_ranku;
        } 
    }
    ranku[0] = inf; 

    //
    // Post Processing : Disperse The Distribution of RankU 
    //
    // if (false)
    if (false && !constraint_task_orders.empty())
    {
        std::vector<bool> orig_visited(list_task.size(), false);
        std::vector<bool> visited(list_task_merged.size(), false);
        std::vector<float> orig_ranku(list_task.size(), 0.0f);

        {
            orig_ranku.resize(num_tasks, 0.0);
            orig_ranku[terminal_node] = fn_compute_avg_cost(terminal_node);
            for (uint i = 1; i < num_tasks; i++) 
            {
                const uint node = list_order[num_tasks - 1 - i];
                const auto& task = list_task[node];
                {
                    if (metric == RankMetric::RankMetricMEAN) 
                    {
                        double max_successor_ranku = -1.0f;
                        for (uint j = 0; j < list_task[node].successors.size(); j++) 
                        {
                            uint succnode = list_task[node].successors[j];
                            double val = list_avg_weight[node][j] + ranku[succnode];
                            max_successor_ranku = max_scalar(max_successor_ranku, val);
                        }
                        
                        if (max_successor_ranku < 0.f) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << max_successor_ranku << "}\n";
                        orig_ranku[node] = fn_compute_avg_cost(node) + max_successor_ranku;
                    } 
                }
            }
            orig_ranku[root_node] = inf; 
        }
        const uint constraint_task_num = constraint_task_orders.size(); orig_visited[constraint_task_orders.back()] = true;
        for (uint i = 1; i < constraint_task_num; i++)
        {
            const uint curr_tid = constraint_task_orders[constraint_task_num - 1 - i];
            const uint next_tid = constraint_task_orders[constraint_task_num     - i];
            if (metric == RankMetric::RankMetricMEAN) 
            {
                orig_ranku[curr_tid] = fn_compute_avg_cost(curr_tid) + list_avg_weight[curr_tid][0] + ranku[next_tid];;
                orig_visited[curr_tid] = true;
            } 
        }
        for (uint i = 1; i < num_tasks; i++) 
        {
            const uint node = list_order[num_tasks - 1 - i];
            if (!orig_visited[node])
            {
                if (metric == RankMetric::RankMetricMEAN) 
                {
                    double max_successor_ranku = -1.0f;
                    for (uint j = 0; j < list_task[node].successors.size(); j++) 
                    {
                        uint succnode = list_task[node].successors[j];
                        double val = list_avg_weight[node][j] + orig_ranku[succnode];
                        max_successor_ranku = max_scalar(max_successor_ranku, val);
                    }
                    if (max_successor_ranku < 0.f) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << max_successor_ranku << "}\n";
                    orig_ranku[node] = fn_compute_avg_cost(node) + max_successor_ranku;
                } 
            }
        }
        orig_ranku[root_node] = inf; 

        for (uint merged_tid = 0; merged_tid < num_tasks_merged; merged_tid++) 
        {
            const auto& merged_task = list_task_merged[merged_tid]; 
            const uint first_tid = merged_task.tasks.front();
            if (orig_visited[first_tid])
            {
                ranku[merged_tid] = orig_ranku[first_tid];
                visited[merged_tid] = true;
            }
        }

        for (uint i = 1; i < num_tasks_merged; i++) 
        {
            const uint merged_tid = num_tasks_merged - 1 - i;
            if (!visited[merged_tid])
            {
                const auto& merged_task = list_task_merged[merged_tid]; 
                const uint first_tid = merged_task.tasks.front();
                double max_successor_ranku = -1.0f;
                for (const uint succ : merged_task.successors) 
                {
                    double val = avgCommunicationCost + ranku[succ]; // avg_weight
                    if (ranku[succ] == 0.0) fast_print_err("Access Zero RankU From Successors!!!!!"); 
                    max_successor_ranku = max_scalar(max_successor_ranku, val);
                }
                ranku[merged_tid] = list_avg_costs_merged[merged_tid] + max_successor_ranku;
            }
        }
        ranku[0] = inf; 
    }
    

}

#else

{
    ranku.resize(num_tasks, 0.0);

    if (metric == RankMetricOCT) 
    {
        oct.resize(num_tasks, std::vector<double>(num_procs, 0.0));
        for (double& value : oct[terminal_node])  value = 0.0;
        for (double& value : oct[root_node])      value = inf;
    }
    ranku[terminal_node] = fn_compute_avg_cost(terminal_node);

    for (uint i = 1; i < num_tasks; i++) 
    {
        const uint node = list_order[num_tasks - 1 - i];
        const auto& task = list_task[node];
        
        //
        // For Each Task From [ Right To Left ]
        //
        {
            if (metric == RankMetric::RankMetricOCT) 
            {
                const uint ti = node;
                double sum_oct = 0.0;
                for (uint pk = 0; pk < num_procs; pk++) 
                {
                    double curr_oct = -1.0f;
                    for (uint j = 0; j < task.successors.size(); j++) 
                    {
                        uint tj = task.successors[j];
                        double min_processor_oct = inf;
                        for (uint pw = 0; pw < num_procs; pw++) 
                        {
                            // if (oct[tj][pw] == 0 && tj != terminal_node) fast_format_err("Get Zero OCT In {} From Successors {}", taskNames.at(task.func_id), taskNames.at(list_task[tj].func_id));
                            double avg_cij = pk == pw ? 0.0 : list_avg_weight[ti][j];
                            double val = oct[tj][pw] + computation_matrix[tj][pw] + avg_cij;
                            min_processor_oct = min_scalar(min_processor_oct, val);
                        }
                        curr_oct = max_scalar(curr_oct, min_processor_oct);
                    }
                    oct[ti][pk] = curr_oct;
                    sum_oct += curr_oct;
                }
                ranku[node] = sum_oct / double(num_procs);
            } 
            else if (metric == RankMetric::RankMetricMEAN) 
            {
                double max_successor_ranku = -1.0f;
                /// 后继节点的ranku + 通信成本
                for (uint j = 0; j < list_task[node].successors.size(); j++) 
                {
                    uint succnode = list_task[node].successors[j];
                    double val = list_avg_weight[node][j] + ranku[succnode];
                    max_successor_ranku = max_scalar(max_successor_ranku, val);
                }
                
                if (max_successor_ranku < 0.f) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << max_successor_ranku << "}\n";
                ranku[node] = fn_compute_avg_cost(node) + max_successor_ranku;
                
                // if(list_task[node].func_id == id_additional_root || list_task[node].func_id == id_update_edge_aabb_dcd){
                    // fast_print(taskNames.at(list_task[node].func_id), fn_compute_avg_cost(node), max_successor_ranku);
                // }

                /// 8 : ranku = 44.33333333333333 , avgcomp_matrix_masked = 16.666666666666668 , max_successor_ranku = 27.666666666666664
                /// 7 : ranku = 35.666666666666664 , avgcomp_matrix_masked = 10.0 , max_successor_ranku = 25.666666666666664
                /// 6 : ranku = 42.666666666666664 , avgcomp_matrix_masked = 11.0 , max_successor_ranku = 31.666666666666664
                /// 1 : ranku = 77.0 , avgcomp_matrix_masked = 16.666666666666668 , max_successor_ranku = 60.33333333333333
                /// 3 : ranku = 80.0 , avgcomp_matrix_masked = 12.666666666666666 , max_successor_ranku = 67.33333333333333
                /// 4 : ranku = 69.0 , avgcomp_matrix_masked = 11.666666666666666 , max_successor_ranku = 57.33333333333333
                /// 5 : ranku = 63.33333333333333 , avgcomp_matrix_masked = 12.666666666666666 , max_successor_ranku = 50.666666666666664
                /// 2 : ranku = 79.99999999999999 , avgcomp_matrix_masked = 14.333333333333334 , max_successor_ranku = 65.66666666666666
                /// 0 : ranku = 108.0 , avgcomp_matrix_masked = 13.0 , max_successor_ranku = 95.0

                // std::cout << "ranku of "  << node  << " : " << ranku[node] << std::endl;
                // std::cout << node << " : ranku = " << ranku[node] << " , avgcomp_matrix_masked = " << fn_compute_avg_cost(node) << " , max_successor_ranku = " << max_successor_ranku << std::endl;
            } 
            // Does Not Work...
            else if (metric == RankMetric::RankMetricWORST) 
            {
                
                float max_cost;
                uint max_cost_proc;
                auto tmp = fn_compute_max_cost(node);
                max_cost_proc = tmp.first;
                max_cost = tmp.second;
                float max_successor_ranku = -1.0f;
                for (uint j = 0; j < list_task[node].successors.size(); j++) 
                {
                    uint succnode = list_task[node].successors[j];
                    float max_succ_cost;
                    auto tmp_succ = fn_compute_max_cost(succnode);
                    max_succ_cost = tmp_succ.second;
                    float val = max_succ_cost + ranku[succnode];
                    max_successor_ranku = std::max(max_successor_ranku, val);
                    
                }
                if (max_successor_ranku < 0.f) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << max_successor_ranku << "}\n";
                ranku[node] = max_cost + max_successor_ranku;
            } 
            // Does Not Work...
            else if (metric == RankMetric::RankMetricBEST) 
            {
                float min_cost;
                uint min_cost_proc;
                auto tmp = fn_compute_min_cost(node);
                min_cost_proc = tmp.first;
                min_cost = tmp.second;
                float min_successor_ranku = inf;
                for (uint j = 0; j < list_task[node].successors.size(); j++) 
                {
                    uint succnode = list_task[node].successors[j];
                    float min_succ_cost;
                    auto tmp_succ = fn_compute_min_cost(succnode);
                    min_succ_cost = tmp_succ.second;
                    float val = min_succ_cost + ranku[succnode];
                    min_successor_ranku = std::min(min_successor_ranku, val);
                }
                if(min_successor_ranku < 0.f || min_successor_ranku == inf) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << min_successor_ranku << "}\n";
                ranku[node] = min_cost + min_successor_ranku;
            }
        }
    }

    ranku[root_node] = inf; 


    //
    // Post Processing : Disperse The Distribution of RankU 
    //
    // if (false)
    if (!constraint_task_orders.empty())
    {
        std::vector<bool> visited(list_task.size(), false);
        
        const uint constraint_task_num = constraint_task_orders.size(); visited[constraint_task_orders.back()] = true;
        for (uint i = 1; i < constraint_task_num; i++)
        {
            const uint curr_tid = constraint_task_orders[constraint_task_num - 1 - i];
            const uint next_tid = constraint_task_orders[constraint_task_num     - i];

            if (metric == RankMetric::RankMetricMEAN) 
            {
                ranku[curr_tid] = fn_compute_avg_cost(curr_tid) + list_avg_weight[curr_tid][0] + ranku[next_tid];;
                visited[curr_tid] = true;
            } 
        }

        // for (auto tid : constraint_task_orders)
        // {
        //     const auto& task = list_task[tid];
        //     fast_format("   Constraint Task {} In Iter {:2} Ranku = {:6.3f} : {}", tid, task.iter_idx, ranku[tid], taskNames.at(task.func_id));
        // }

        for (uint i = 1; i < num_tasks; i++) 
        {
            const uint node = list_order[num_tasks - 1 - i];
            if (!visited[node]) // Skip Re-Sized Task
            {
                if (metric == RankMetric::RankMetricMEAN) 
                {
                    double max_successor_ranku = -1.0f;
                    for (uint j = 0; j < list_task[node].successors.size(); j++) 
                    {
                        uint succnode = list_task[node].successors[j];
                        double val = list_avg_weight[node][j] + ranku[succnode];
                        max_successor_ranku = max_scalar(max_successor_ranku, val);
                    }
                    if (max_successor_ranku < 0.f) std::cerr << "Expected maximum successor ranku to be greater or equal to 0 but was {" << max_successor_ranku << "}\n";
                    ranku[node] = fn_compute_avg_cost(node) + max_successor_ranku;
                } 
            }
        }
        ranku[root_node] = inf; 
    }
    
}
#endif

}

ScheduleEvent fn_get_eft_in_processor(const uint proc, const std::vector<ScheduleEvent>& proc_schedule, 
    const float node, const float ready_time, 
    const float computation_time, const float communication_time_inner_device) 
{
    for (uint idx = 0; idx < proc_schedule.size(); ++idx) 
    {
        const ScheduleEvent& prev_job = proc_schedule[idx];
        if (idx == 0) 
        {
            if (prev_job.start > ready_time + computation_time + communication_time_inner_device) 
            {
                float job_start = ready_time;
                // if (job_start >= inf || computation_time >= inf) { fast_format_err("1 Make inf {} -> {} + {}, ready_time = {}", job_start, job_start, computation_time, ready_time); prev_job.print(); }
                return ScheduleEvent(node, proc, job_start, job_start + computation_time); // EFT = EST + w_ij, Where w_ij is Computation Cost of Task i in Prosessor j
            }
        } 
        if (idx == proc_schedule.size() - 1) 
        {
            float job_start = max_scalar(ready_time, prev_job.end + communication_time_inner_device);
            // if (job_start >= inf || computation_time >= inf) { fast_format_err("2 Make inf {} -> {} + {}, ready_time = {}", job_start, job_start, computation_time, ready_time); prev_job.print(); }
            return ScheduleEvent(node, proc, job_start, job_start + computation_time);
        } 
        else 
        {
            const ScheduleEvent& next_job = proc_schedule[idx + 1];
            float slot_size = next_job.start - max_scalar(ready_time, prev_job.end + communication_time_inner_device);
            if (slot_size > computation_time) 
            {
                float job_start = max_scalar(ready_time, prev_job.end + communication_time_inner_device);
                // if (job_start >= inf || computation_time >= inf) { fast_format_err("3 Make inf {} -> {} + {}, ready_time = {}", job_start, job_start, computation_time, ready_time); prev_job.print(); }
                return ScheduleEvent(node, proc, job_start, job_start + computation_time);
            }
        }
    }
    // if (ready_time >= inf || computation_time >= inf) { fast_format_err("4 Make inf {} -> {} + {}", ready_time, ready_time, computation_time);  }
    return ScheduleEvent(node, proc, ready_time, ready_time + computation_time);
};
void insertAndMergeTask(std::vector<ScheduleEvent>& proc_schedules, const ScheduleEvent& new_task) 
{
    // 初始化插入区间
    float bg = new_task.start;
    float ed = new_task.end;
    uint task_id = new_task.task_id;
    uint proc = new_task.proc;

    // 查找所有与新任务重叠的任务区间
    auto it = proc_schedules.begin();
    while (it != proc_schedules.end()) 
    {
        // 如果当前任务与新任务重叠
        if (!(it->end < bg || it->start > ed)) 
        {
            // 更新合并后的区间
            bg = std::min(bg, it->start);
            ed = std::max(ed, it->end);
            // 删除当前重叠的任务
            it = proc_schedules.erase(it);
        } else {  ++it; }
    }

    // 插入合并后的新任务
    proc_schedules.insert(
        std::upper_bound(proc_schedules.begin(), proc_schedules.end(), ScheduleEvent(task_id, proc, bg, ed)),
        ScheduleEvent(task_id, proc, bg, ed)
    );
};

ScheduleEvent Scheduler::_compute_eft(uint node, uint proc){

    // const float weight = 0.f;
    float ready_time = 0;
    auto& task = list_task[node];
    // auto& offsets = task.list_offset;
    
    // 对于前置的任务
    for (uint i = 0; i < task.predecessors.size(); i++) {
        auto pred_node = task.predecessors[i];
        auto& pred_task = list_task[pred_node];

        const ScheduleEvent& pred_job = task_schedules[pred_node];
        float comm_speed = communication_speed_matrix[pred_job.proc][proc];

        // uint offset = offsets[i];
        uint offset;
        auto find = std::find(pred_task.successors.begin(), pred_task.successors.end(), node);
        if(find == pred_task.successors.end())  { fast_print_err("Can not Find Task!!!"); break;}
        else                                    { offset = find - pred_task.successors.begin(); }
        float weight = pred_task.list_weight[offset];
        float comm_cost;

        /// 5.03124 : 4.96319 & 5.06651
        /// 4.93648 : 4.90397 & 5.09523
        if (weight == 0.f) comm_cost = 0.f;
        else comm_cost = fn_get_communication_cost(proc, pred_node, node);

        float ready_time_t;
        if (comm_speed == 0) {
            ready_time_t = pred_job.end + communication_startup[pred_job.proc];
        } 
        else {
            ready_time_t = pred_job.end + comm_cost + communication_startup[pred_job.proc];
        }
        ready_time = max_scalar(ready_time, ready_time_t);
    }

    // fast_print("Ready time determined to be", ready_time);

    float computation_time = computation_matrix[node][proc];
    const ListSchedule& job_list = proc_schedules[proc];

    for (uint idx = 0; idx < job_list.size(); ++idx) {
        const ScheduleEvent& prev_job = job_list[idx];
        if (idx == 0) {
            if (prev_job.start > ready_time + computation_time) {
                float job_start = ready_time;
                return ScheduleEvent(node, proc, job_start, job_start + computation_time);
            }
        } 
        
        if (idx == job_list.size() - 1) {
            float job_start = max_scalar(ready_time, prev_job.end);
            return ScheduleEvent(node, proc, job_start, job_start + computation_time);
        } 
        else {
            const ScheduleEvent& next_job = job_list[idx + 1];
            float slot_size = next_job.start - max_scalar(ready_time, prev_job.end);
            if (slot_size >= computation_time) {
                float job_start = max_scalar(ready_time, prev_job.end);
                return ScheduleEvent(node, proc, job_start, job_start + computation_time);
            }
        }
    }

    if (ready_time >= inf || computation_time >= inf) {
        fast_print_err("Make inf inf computing EFT...", computation_time);
    }

    /// If the loop completes without finding a suitable slot, schedule the job after the last job
    return ScheduleEvent(node, proc, ready_time, ready_time + computation_time);
}


float Scheduler::fn_get_communication_cost(const uint proc, const uint pred_node, const uint input_node)
{
    const auto& pred_task = list_task[pred_node];
    const ScheduleEvent& pred_job = task_schedules[pred_node];

    // 找到本节点在前置节点的偏移量
    uint offset;
    auto find = std::find(pred_task.successors.begin(), pred_task.successors.end(), input_node);
    if (find == pred_task.successors.end())  { fast_format_err("Can NOT Find Current Task ({}) In Pred's ({}) Succ", taskNames.at(list_task[input_node].func_id), taskNames.at(pred_task.func_id)); exit(0); }
    else                                    { offset = find - pred_task.successors.begin(); }

    float weight = pred_task.list_weight[offset];
    float comm_cost = weight == 0.f ? 0.f 
                                   : !communication_cost_matrix_uma.empty() ? communication_cost_matrix_uma[pred_job.proc][proc] 
                                   : pred_job.proc == proc ? communication_speed_matrix[pred_job.proc][proc] : weight / communication_speed_matrix[pred_job.proc][proc];
    // if (comm_cost >= inf) 
    // {
    //     fast_format_err("Get inf in comm_cost : {} -> {} , weight = {}", pred_node, input_node, weight);
    //     fast_print("communication_cost_matrix_uma"); for (const auto& list : communication_cost_matrix_uma) { for (const auto& comm : list) { fast_print_single(comm); } } fast_print();
    //     fast_print("communication_speed_matrix"); for (const auto& list : communication_speed_matrix) { for (const auto& comm : list) { fast_print_single(comm); } } fast_print();
    // }
    return comm_cost;
};
float Scheduler::fn_get_inner_communication_cost(const uint proc)
{
    return !communication_cost_matrix_uma.empty() ? communication_cost_matrix_uma[proc][proc] : communication_speed_matrix[proc][proc];
}
ScheduleEvent Scheduler::_compute_eft_extend(uint node, uint proc)
{
    float ready_time = 0; // T_Available
    const auto& task = list_task[node];
    const float computation_time = computation_matrix[node][proc];
    const float communication_time_inner_device = fn_get_inner_communication_cost(proc);

    constexpr float reflection_rate = 1;

#if false // Does Not Work...
    const bool use_refliction_computation_time = false; 
    if constexpr (use_refliction_computation_time)
    {
        const uint num_tasks = list_task.size(); const int center = num_tasks / 2;
        std::vector< uint > original_get_sorted(num_tasks);
        std::vector< float > compuatation_rate (num_tasks);
        std::vector< uint  > sorted_get_original(num_tasks);
        for (uint tid = 0; tid < num_tasks; tid++) 
        {
            const std::vector<float>& curr_cost = computation_matrix[tid];
            float sum_cost = max_scalar(Epsilon, curr_cost[0] + curr_cost[1]);
            // float sum_cost = max_scalar(Epsilon, std::accumulate(curr_cost.begin(), curr_cost.end(), 0.0f));
            compuatation_rate[tid] = (curr_cost[proc] / sum_cost);
            sorted_get_original[tid] = tid;
        }
        std::sort(sorted_get_original.begin(), sorted_get_original.end(), 
            [&compuatation_rate](uint left, uint right)
            { return compuatation_rate[left] < compuatation_rate[right]; }); // Least Order
        for (uint sorted_tid = 0; sorted_tid < num_tasks; sorted_tid++) 
            {   uint orig_tid = sorted_get_original[sorted_tid]; original_get_sorted[orig_tid] = sorted_tid;  
                float sorted_rate = compuatation_rate[orig_tid]; if (node == 0) fast_print_single(sorted_rate);
            }
        
        const int sorted_idx = original_get_sorted[node];
        reflection_rate = 1.0f + 0.5f * float(sorted_idx - center) / float(num_tasks); // -0.5 < (sorted_idx - center)/(num_tasks) < 0.5
    }
#endif

    auto fn_get_ready_time_from_pred = [&](const uint pred_node, const uint input_node)
    {
        const ScheduleEvent& pred_job = task_schedules[pred_node];

        float comm_cost = fn_get_communication_cost(proc, pred_node, input_node);

        float ready_time_t = 
                        //    comm_speed == 0 ? pred_job.end              // 设备不变
                        //                      : 
                                            pred_job.end + comm_cost; // 加上通信成本
        return ready_time_t;
    };

    // else 
    {
        // Can not ealier than its predecessors => Same as original plan
        for (const uint& pred_node : task.predecessors) 
        {
            // EST(n_i, p_j) = max{ T_available, max( AFT(n_m) + c_mi) } , p_j in Processors, n_m in Predecessors of n_i
            float ready_time_t = fn_get_ready_time_from_pred(pred_node, node);
            ready_time = std::max(ready_time, ready_time_t);
        }
    }
    
    // ListSchedule copy_proc_schedules = proc_schedules[proc];
    // fast_format("Task schedule of node {}, proc = {} , proc_schedule_size = {}, ready_time = {}, computation_time = {}, communication_time_inner_device = {}, eft = {}", 
    //     node, proc, proc_schedules[proc].size(), ready_time, computation_time, communication_time_inner_device, fn_get_eft_in_processor(proc, proc_schedules[proc], node, ready_time, computation_time, communication_time_inner_device).end);
    return fn_get_eft_in_processor(proc, proc_schedules[proc], node, ready_time, computation_time, communication_time_inner_device);

}
ScheduleEvent Scheduler::_compute_eft_merged(uint node, uint proc)
{

    float ready_time = 0; // T_Available
    const auto& task = list_task_merged[node];
    const float computation_time = task.costs[proc];
    const float communication_time_inner_device = fn_get_inner_communication_cost(proc);

    auto fn_update_ready_time_from_pred_of = [&](const uint pred_node, const uint input_node)
    {
        const auto& pred_task = list_task_merged[pred_node];
        const ScheduleEvent& pred_job = task_schedules_merged[pred_node];
        const float comm_speed = communication_speed_matrix[pred_job.proc][proc];

        uint offset;
        auto find = std::find(pred_task.successors.begin(), pred_task.successors.end(), input_node);
        if (find == pred_task.successors.end())  { fast_print_err("Can NOT Find Task!!!"); exit(0);}
        else                                     { offset = find - pred_task.successors.begin(); }

        float weight = pred_task.weights[offset];
        float comm_cost = weight == 0.f ? 0.f 
                                        : !communication_cost_matrix_uma.empty() ? communication_cost_matrix_uma[pred_job.proc][proc] 
                                                                                 : weight / comm_speed;
        float ready_time_t = pred_job.end + comm_cost; // 加上通信成本
        ready_time = max_scalar(ready_time, ready_time_t);
    };

    {
        for (const uint& pred_node : task.predecessors) 
        {
            fn_update_ready_time_from_pred_of(pred_node, node);
        }
    }

    ListSchedule copy_proc_schedules = proc_schedules_merged[proc];

    return fn_get_eft_in_processor(proc, copy_proc_schedules, node, ready_time, computation_time, communication_time_inner_device);

}

void Scheduler::scheduler_dag() 
{
    const uint num_procs = communication_startup.size();
    
#if USE_MERGED_TASK
    const uint num_tasks = list_task.size();
    
    compute_ranku(num_procs); // => Get The Merged Task
    const uint num_tasks_merged = list_task_merged.size();

    std::vector<uint> sorted_nodes(num_tasks_merged);
    for (uint tid = 0; tid < num_tasks_merged; tid++) {  sorted_nodes[tid] = tid; }  
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), [&](const uint tid1, const uint tid2) 
    { 
        return ranku[tid1] > ranku[tid2];  // Reverse Order
    });

    // Check Sort By RankU
    {
        std::vector<uint> original_get_sorted(num_tasks_merged);
        for (uint sortedIdx = 0; sortedIdx < num_tasks_merged; sortedIdx++){
            const uint origIdx = sorted_nodes[sortedIdx];
            original_get_sorted[origIdx] = sortedIdx;
        }  
        for (uint sortedIdx = 0; sortedIdx < sorted_nodes.size(); sortedIdx++) {
            const uint merged_tid = sorted_nodes[sortedIdx];
            auto& merged_task = list_task_merged[merged_tid];
            // fast_format("Merged Task {:2} : {:6.3f}", merged_tid, ranku[merged_tid]); 
            // for (auto tid : merged_task.tasks) fast_format("            {:2} : {}", list_task[tid].iter_idx, taskNames.at(list_task[tid].func_id)); 
            for (auto pred : merged_task.predecessors) { if (original_get_sorted[pred] >= sortedIdx) fast_print_err("Miss The Connection RelationShip"); }
            for (auto succ : merged_task.successors)   { if (original_get_sorted[succ] <= sortedIdx) fast_print_err("Miss The Connection RelationShip"); }
        }
    }  

    // sorted_nodes = topological_sort_DFS(list_task_merged);

    task_schedules.resize(num_tasks);
    task_schedules_merged.resize(num_tasks_merged);
    proc_schedules.resize(num_procs);
    proc_schedules_merged.resize(num_procs);

    {
        for (uint merged_tid : sorted_nodes) {

            const auto& task = list_task_merged[merged_tid];

            ScheduleEvent minTaskSchedule(merged_tid, 0, inf, inf); 
            float minEDP = inf;
            OpMode op_mode = OpMode::OpModeEFT;

            if (op_mode == OpMode::OpModeEFT) {
                for (uint proc = 0; proc < num_procs; ++proc) {
                    ScheduleEvent taskschedule = _compute_eft_merged(merged_tid, proc, is_uniform_memory_architecture);
                    if (taskschedule.start >= inf) {
                        fast_print_err("Make Inf..."); return;
                    }
                    if (taskschedule.end < minTaskSchedule.end) {
                        minTaskSchedule = taskschedule;
                    }     
                }
            }

            const uint selected_proc = minTaskSchedule.proc;
            const float communication_cost_in_selected_proc = communication_cost_matrix_uma[selected_proc][selected_proc];
            DeviceType type;
            if (selected_proc)      type = DeviceTypeCpu;
            else if (selected_proc) type = DeviceTypeGpu;
            for (const uint inner_tid : task.tasks) {
                const auto& task = list_task[inner_tid];
                if (!task.has_implementation(type)) fast_print_err("Does Not Find Implementation of Task", taskNames.at(task.func_id));
            }
            
            // Insert Into Task Schedule List & Processor Schedule List
            task_schedules_merged[merged_tid] = minTaskSchedule;
            proc_schedules_merged[selected_proc].push_back(minTaskSchedule);
            std::sort(proc_schedules_merged[selected_proc].begin(), proc_schedules_merged[selected_proc].end());   

            // Fill-in Inner Task Schedule
            float prev_end = minTaskSchedule.start;
            for (const uint tid : task.tasks) {
                const float computation_cost = computation_matrix[tid][selected_proc];

                ScheduleEvent currTaskSchedule(tid, selected_proc, prev_end, prev_end + computation_cost); 
                task_schedules[tid] = currTaskSchedule;
                proc_schedules[selected_proc].push_back(currTaskSchedule);
                std::sort(proc_schedules[selected_proc].begin(), proc_schedules[selected_proc].end());   

                prev_end = prev_end + computation_cost + communication_cost_in_selected_proc;
            }
            
        }
    }
    
    fast_format("End Time By Merging Tasks = {:6.3f}&{:6.3f} / {:6.3f}&{:6.3f}", 
        proc_schedules_merged[0].back().end, proc_schedules_merged[1].back().end,
        proc_schedules[0].back().end, proc_schedules[1].back().end);

#else

    //
    // HEFT : For v Tasks, p Processors => Complexity is O(v^2 p) 
    //

    // standardizing_dag();

    const uint num_tasks = list_task.size();
    
    ///
    /// Sort the nodes based on Rank-U in descending order
    ///
    // fast_print("\n================ Performing Rank-U Computation ================\n");
    compute_ranku(num_procs); 
    const bool use_peft = !oct.empty();

    // fast_print("\n================ Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ===============\n");
    
    ///
    /// Sort by ranku
    ///
    // std::vector<uint> sorted_nodes(num_tasks);
    sorted_nodes.resize(num_tasks);
    for (uint tid = 0; tid < num_tasks; tid++){ sorted_nodes[tid] = tid; }  
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), [&](const uint& tid1, const uint& tid2) 
    { 
        // reverse=True
        return ranku[tid1] > ranku[tid2];  
    });

    // if constexpr (false) // Check Sort
    {
        std::vector<uint> original_get_sorted(num_tasks);
        for (uint sorted_idx = 0 ; sorted_idx < num_tasks; sorted_idx++)
        {
            const uint orig_idx = sorted_nodes[sorted_idx];
            original_get_sorted[orig_idx] = sorted_idx;
        }

        // Must Check, If not, Culculating EFT Will Miss The Connection
        for (uint sorted_idx = 0; sorted_idx < num_tasks; sorted_idx++)
        {
            const uint orig_idx = sorted_nodes[sorted_idx]; 
            const auto& task = list_task[orig_idx];
            for (const uint pred : task.predecessors)
            {
                const uint sorted_pred = original_get_sorted[pred];
                if (sorted_pred >= sorted_idx)
                {
                    fast_format_err("Sorted Task {:2}->{:2}({}, ranku={}) Is Ealier Than Its Pred {:2}->{:2}({}, ranku={}) ", 
                        orig_idx, sorted_idx, taskNames.at(task.func_id), ranku[orig_idx],
                        pred, sorted_pred, taskNames.at(list_task[pred].func_id), ranku[pred]);
                    fast_format_err("Might Be Ranku Miss The Relationship");
                    // fast_print_single("         Its Preds : ");  for (const uint pp : task.predecessors)
                    // {
                    //     fast_format_single(" {}", taskNames.at(list_task[pp].func_id));
                    // }
                    // fast_print_single("\n         Its Succs : ");  for (const uint pp : task.successors)
                    // {
                    //     fast_format_single(" {}", taskNames.at(list_task[pp].func_id));
                    // } fast_print();
                }
            }
        }
    }
    

    /// node : 0 , ranku = 108.0
    /// node : 3 , ranku = 80.0
    /// node : 2 , ranku = 79.99999999999999
    /// node : 1 , ranku = 77.0
    /// node : 4 , ranku = 69.0
    /// node : 5 , ranku = 63.33333333333333
    /// node : 8 , ranku = 44.33333333333333
    /// node : 6 , ranku = 42.666666666666664
    /// node : 7 , ranku = 35.666666666666664
    /// node : 9 , ranku = 14.666666666666666

    const bool print_ranku = bool_print_scheduling_datail;
    if (print_ranku) {
        // fast_print("Sort by Topology Sort");
        // for (auto tid : list_order) {
        //     auto& task = list_task[tid];
        //     fast_format(" {:6.3f} : [Iter {:2}, Color {:2}] => {}", ranku[tid], task.iter_idx, task.cluster_idx, Launcher::taskNames.at(task.func_id) );
        // }
        fast_print("Sort by Ranku");
        for (auto tid : sorted_nodes) {
            auto& task = list_task[tid];
            if (task.iter_idx == -1u) {
                fast_format(" {:6.3f} : [Iter {:2}, Cluter {:2}] => {}", ranku[tid], tid, task.cluster_idx, Launcher::taskNames.at(task.func_id) );
            }
            else {
                fast_format(" {:6.3f} : [Tid {:2}, Iter {:2}, Color {:2}] => {}", ranku[tid], tid, task.iter_idx, task.cluster_idx, Launcher::taskNames.at(task.func_id) );
            }
            // std::cout << "  " << ranku[tid] << " : " << "[" << task.cluster_idx << "] " << Launcher::taskNames.at(task.func_id) << " \n";
            // std::cout << "node : " << tid << " , rank u = " << ranku[tid] << " \n";
        }
    }
    
    if (sorted_nodes[0] != root_node) {
        fast_print("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n");
        uint idx = std::distance(sorted_nodes.begin(), std::find(sorted_nodes.begin(), sorted_nodes.end(), root_node));
        std::swap(sorted_nodes[0], sorted_nodes[idx]);
    }


    // Schedule tasks based on Rank-U
    task_schedules.resize(num_tasks);
    proc_schedules.resize(num_procs);

    for (uint node : sorted_nodes) {

        const auto& task = list_task[node];    
        ScheduleEvent minTaskSchedule(node, 0, inf, inf); 
        float minOptimisticCost = inf;
        float minEDP = inf;
        OpMode op_mode = OpMode::OpModeEFT;

        if (op_mode == OpMode::OpModeEFT){
            for (uint proc = 0; proc < num_procs; ++proc) {
                // ScheduleEvent taskschedule = _compute_eft(node, proc);
                ScheduleEvent taskschedule = _compute_eft_extend(node, proc);

                /// For node 0 on processor 0, the EFT is ScheduleEvent(task=0, start=0, end=14, proc=0)
                /// For node 0 on processor 1, the EFT is ScheduleEvent(task=0, start=0, end=16, proc=1)
                /// For node 0 on processor 2, the EFT is ScheduleEvent(task=0, start=0, end=9, proc=2)
                if (bool_print_scheduling_datail) taskschedule.print();
                if (taskschedule.start >= inf) {
                    fast_print_err("Make Inf...");
                    return;
                }
                if (!use_peft) {
                    if (taskschedule.end < minTaskSchedule.end) {
                        minTaskSchedule = taskschedule;
                    }
                }
                else {
                    if (taskschedule.end + oct[node][proc] < minTaskSchedule.end + minOptimisticCost) {
                        minTaskSchedule = taskschedule;
                        minOptimisticCost = oct[node][proc];
                    }
                }
            }
        }
        // else if(op_mode == OpMode::OpModeEDP_ABS)
        // else if(op_mode == OpMode::OpModeEDP_REL)
        // else if(op_mode == OpMode::OpModeENERGY)
        ///    Following Methods Requres Energy Dictionary

        /// node : 0 , minTaskSchedule = ScheduleEvent(task=0, start=0, end=9, proc=2) 
        /// node : 3 , minTaskSchedule = ScheduleEvent(task=3, start=18.0, end=26.0, proc=1) 
        /// node : 2 , minTaskSchedule = ScheduleEvent(task=2, start=9, end=28, proc=2) 
        /// node : 1 , minTaskSchedule = ScheduleEvent(task=1, start=27.0, end=40.0, proc=0) 
        /// node : 4 , minTaskSchedule = ScheduleEvent(task=4, start=28, end=38, proc=2) 
        /// node : 5 , minTaskSchedule = ScheduleEvent(task=5, start=26.0, end=42.0, proc=1) 
        /// node : 8 , minTaskSchedule = ScheduleEvent(task=8, start=56.0, end=68.0, proc=1) 
        /// node : 6 , minTaskSchedule = ScheduleEvent(task=6, start=38, end=49, proc=2) 
        /// node : 7 , minTaskSchedule = ScheduleEvent(task=7, start=57.0, end=62.0, proc=0) 
        /// node : 9 , minTaskSchedule = ScheduleEvent(task=9, start=73.0, end=80.0, proc=1)

        // minTaskSchedule.print();

        // DeviceType type;
        // if (minTaskSchedule.proc == 0)       type = DeviceTypeCpu;
        // else if (minTaskSchedule.proc == 1)  type = DeviceTypeGpu;
        // if (!task.has_implementation(type)) fast_print_err("Does Not Find Implementation of Task", taskNames.at(task.func_id));
        
        ListSchedule& curr_proc_schedules = proc_schedules[minTaskSchedule.proc];
        task_schedules[node] = minTaskSchedule;
        curr_proc_schedules.push_back(minTaskSchedule);

        std::sort(curr_proc_schedules.begin(), curr_proc_schedules.end(), 
        [&](const ScheduleEvent& a, const ScheduleEvent& b)
        {
            // if(a.start == b.start) std::cerr << "Two tasks have the same start time \n";
            return a < b;
        });   

        // fn_insert_un_updated(minTaskSchedule, task);
        // auto insert_sync_cloth_schedule = fn_insert_sync_task(minTaskSchedule, task, ConstraintTaskTypeCloth);
        // auto insert_sync_tet_schedule = fn_insert_sync_task(insert_sync_cloth_schedule, task, ConstraintTaskTypeTet);

    }

    /// Processor 0 has the following jobs:
    ///   [ScheduleEvent(task=1, start=27.0, end=40.0, proc=0), 
    ///    ScheduleEvent(task=7, start=57.0, end=62.0, proc=0)]
    /// Processor 1 has the following jobs:
    ///   [ScheduleEvent(task=3, start=18.0, end=26.0, proc=1), 
    ///    ScheduleEvent(task=5, start=26.0, end=42.0, proc=1), 
    ///    ScheduleEvent(task=8, start=56.0, end=68.0, proc=1), 
    ///    ScheduleEvent(task=9, start=73.0, end=80.0, proc=1)]
    /// Processor 2 has the following jobs:
    ///   [ScheduleEvent(task=0, start=0, end=9, proc=2), 
    ///    ScheduleEvent(task=2, start=9, end=28, proc=2), 
    ///    ScheduleEvent(task=4, start=28, end=38, proc=2), 
    ///    ScheduleEvent(task=6, start=38, end=49, proc=2)]
    
    // print_schedule(num_procs);
    // print_schedule_to_graph(num_procs);



#endif

    //
    // Post processing: Add communication between iteractive tasks for asynchronous iteration
    //
    // if constexpr (false)
    if (!constraint_task_orders.empty() && !communication_cost_matrix_uma.empty()) // For iteration tasks, make additional connections
    {
        auto fn_is_cloth_task = [](const Task& task)
        {
            return 
                task.func_id == 
                id_xpbd_constraint_stretch_mass_spring 
                || task.func_id == id_xpbd_constraint_stretch_mass_spring_half 
                || task.func_id == id_xpbd_constraint_stretch_fem 
                || task.func_id == id_xpbd_constraint_stretch_fem_half 
                || task.func_id == id_xpbd_constraint_bending 
                || task.func_id == id_xpbd_constraint_bending_half 
                || task.func_id == id_xpbd_constraint_obstacle_collision_vv_cloth 
                || task.func_id == id_xpbd_constraint_ground_collision_cloth 
                || task.func_id == id_xpbd_constraint_self_collision_vv_cloth
                || task.func_id == id_xpbd_constraint_self_collision_vv_half_cloth 
                || task.func_id == id_xpbd_constraint_self_collision_vv_cross
                || task.func_id == id_xpbd_constraint_self_collision_vv_half_cross
                || task.func_id == id_xpbd_constraint_last_node

                || task.func_id == id_vbd_evaluate_inertia
                || task.func_id == id_vbd_evaluate_stretch_mass_spring
                || task.func_id == id_vbd_evaluate_stretch_fem
                || task.func_id == id_vbd_evaluate_bending
                || task.func_id == id_vbd_evaluate_ground_collision
                || task.func_id == id_vbd_evaluate_obstacle_collision
                || task.func_id == id_vbd_evaluate_self_collision
                || task.func_id == id_vbd_step

                || task.func_id == id_vbd_all_in_one
                || task.func_id == id_sod_all_in_one
                ;
        };
        auto fn_is_tet_task = [](const Task& task)
        {
            return 
                task.func_id == 
                id_xpbd_constraint_stress 
                || task.func_id == id_xpbd_constraint_stress_half 
                || task.func_id == id_xpbd_constraint_obstacle_collision_vv_tet 
                || task.func_id == id_xpbd_constraint_ground_collision_tet 
                || task.func_id == id_xpbd_constraint_self_collision_vv_tet 
                || task.func_id == id_xpbd_constraint_self_collision_vv_half_tet 
                || task.func_id == id_xpbd_constraint_self_collision_vv_cross 
                || task.func_id == id_xpbd_constraint_self_collision_vv_half_cross 
                || task.func_id == id_xpbd_constraint_last_node

                || task.func_id == id_vbd_all_in_one
                || task.func_id == id_sod_all_in_one
                ;
        };
        
        // Set Connection
        std::vector<bool> is_constraint_task(num_tasks, false);
        std::vector<bool> is_producer_task(num_tasks, false);
        std::vector<bool> is_cloth_constraint_task(num_tasks, false);
        std::vector<bool> is_tet_constraint_task(num_tasks, false);
        uint predict_position_tid = -1u;

        std::vector<uint> constraint_tasks;
        for (uint tid = 0; tid < num_tasks; tid++)
        {
            const auto& task = list_task[tid];
            if (
                (task.func_id >= id_xpbd_constraint_stretch_mass_spring && task.func_id <= id_xpbd_constraint_last_node)
                || (task.func_id >= id_vbd_evaluate_inertia && task.func_id <= id_vbd_all_in_one)
                || (task.func_id >= id_sod_evaluate_energy_gauss_seidel && task.func_id <= id_sod_all_in_one)
                )
            {
                is_constraint_task[tid] = true;
            }
            if (
                (task.func_id >= id_xpbd_constraint_stretch_mass_spring && task.func_id <= id_xpbd_constraint_last_node)
                || (task.func_id >= id_vbd_step && task.func_id <= id_vbd_all_in_one)
                || (task.func_id >= id_sod_step && task.func_id <= id_sod_all_in_one)
                )
            {
                is_producer_task[tid] = true;
            }
            if (fn_is_cloth_task(task)) is_cloth_constraint_task[tid] = true;
            if (fn_is_tet_task(task))   is_tet_constraint_task[tid] = true;
            if (task.func_id == id_xpbd_predict_position) predict_position_tid = tid;
        }

        // Find The Shortest Connection Between Devices
        std::vector<std::vector<uint>> list_in(num_tasks);
        std::vector<std::vector<uint>> list_out(num_tasks);
        for (uint proc = 0; proc < num_procs; proc++)
        {
            const auto& proc_schedule = proc_schedules[proc];
            const auto& another_proc_schedule = proc_schedules[1 - proc];
            const float comm_time = communication_cost_matrix_uma[proc][1 - proc];
            for (uint i = 0; i < proc_schedule.size(); i++)
            {
                const auto& self_schedule = proc_schedule[i];
                const uint tid = self_schedule.task_id;
                // if (is_producer_task[tid])
                if (is_constraint_task[tid])
                {
                    const bool is_cloth = is_cloth_constraint_task[tid];
                    const bool is_tet = is_tet_constraint_task[tid];
                    
                    for (uint j = 0; j < another_proc_schedule.size(); j++)
                    {
                        const auto& another_schedule = another_proc_schedule[j];
                        if (is_constraint_task[another_schedule.task_id] && self_schedule.end + comm_time <= another_schedule.start)
                        {
                            const uint adj_tid = another_schedule.task_id;
                            const bool adj_is_cloth = is_cloth_constraint_task[adj_tid];
                            const bool adj_is_tet = is_tet_constraint_task[adj_tid];
                            if ( (is_cloth && adj_is_cloth) || (is_tet && adj_is_tet) )
                            {
                                list_out[tid] = {adj_tid};
                                list_in[adj_tid].push_back(tid);
                                break;
                            }
                        }
                    }
                }
            }
        }

    #define USE_MAIN_DEVICE true
    #define MAIN_DEVICE_ID 1

        // Merge Redundant Connection
    #if USE_MAIN_DEVICE
        {
            // NOT The Main Device: Only Keep The Last Input
            for (uint proc = 0; proc < num_procs; proc++)
            {
                if (proc == MAIN_DEVICE_ID) continue;
                const auto& proc_schedule = proc_schedules[proc];
                for (uint i = 0; i < proc_schedule.size(); i++)
                {
                    const auto& self_schedule = proc_schedule[i];
                    const uint tid = self_schedule.task_id;
                    if (is_constraint_task[tid])
                    {
                        list_task[tid].is_allocated_to_main_device = false;
                        auto& ins = list_in[tid];
                        if (ins.size() > 1)
                        {
                            const uint last_in = ins.back();
                            for (uint j = 0; j < ins.size() - 1; j++) // Drop the connection to previous tasks
                            {
                                const uint input_node = ins[j];
                                auto& outs = list_out[input_node];
                                outs.erase(std::find(outs.begin(), outs.end(), tid));
                            }
                            ins = {last_in};
                        }
                        if (!ins.empty())
                        {
                            set_connect(ins.back(), tid);
                        }
                    }
                }
            }
            // For Main Device: Keep All Inputs
            const uint main_proc = MAIN_DEVICE_ID;
            const auto& main_proc_schedule = proc_schedules[main_proc];
            for (uint i = 0; i < main_proc_schedule.size(); i++)
            {
                const auto& self_schedule = main_proc_schedule[i];
                const uint tid = self_schedule.task_id;
                if (is_constraint_task[tid])
                {
                    auto& ins = list_in[tid];
                    if (ins.size() > 1)
                    {
                        list_task[tid].is_allocated_to_main_device = true;
                        std::vector<uint> new_ins;
                        for (uint j = 0; j < ins.size() - 1; j++)
                        {
                            const uint input_node = ins[j];
                            const uint next_node = ins[j + 1];
                            if (list_in[next_node].empty())
                            {
                                auto& outs = list_out[input_node];
                                outs.erase(std::find(outs.begin(), outs.end(), tid));
                            }
                            else 
                            {
                                new_ins.push_back(input_node);
                            }
                        } new_ins.push_back(ins.back());
                        ins = new_ins;
                    }
                    if (!ins.empty())
                    {
                        for (const uint input_node : ins) set_connect(input_node, tid);
                    }
                }
            }
        }
    #else
        for (uint tid = 0; tid < num_tasks; tid++)
        {
            if (is_constraint_task[tid])
            {
                auto& ins = list_in[tid];
                if (ins.size() > 1)
                {
                    const uint last_in = ins.back();
                    for (uint j = 0; j < ins.size() - 1; j++)
                    {
                        const uint input_node = ins[j];
                        auto& outs = list_out[input_node];
                        outs.erase(std::find(outs.begin(), outs.end(), tid));
                    }
                    ins = {last_in};
                }
                if (!ins.empty())
                {
                    set_connect(ins.back(), tid);
                    // fast_format("  Merged Connection {} -> {} ( {} -> {} )", 
                    //     ins.back(), tid, 
                    //     taskNames.at(list_task[ins.back()].func_id), taskNames.at(list_task[tid].func_id));
                }
            }
        }
    #endif

        uint buffer_idx = 0;
        auto fn_update_buffer_idx = [&buffer_idx]() { return buffer_idx++; };
        std::vector<uint> task_buffers(num_tasks, -1u);
    
        // Traverse Schedules
        {
            uint index_cpu = 0; 
            uint index_gpu = 0; 
            std::vector<uint> left_constraint_tid(num_procs, -1u);

            const auto& cpu_schedule = proc_schedules[0];
            const auto& gpu_schedule = proc_schedules[1];

            if (!cpu_schedule.empty()) list_task[cpu_schedule.front().task_id].is_first_iterative_task = true;
            if (!gpu_schedule.empty()) list_task[gpu_schedule.front().task_id].is_first_iterative_task = true;
            
            while (index_cpu < cpu_schedule.size() || index_gpu < gpu_schedule.size()) 
            {
                const ScheduleEvent* next_task = nullptr;
                uint min_proc = -1u; float min_start = inf - 1;
                uint min_offset = -1u;

                if (index_cpu < cpu_schedule.size() && (index_gpu >= gpu_schedule.size() || 
                    cpu_schedule[index_cpu].start <= gpu_schedule[index_gpu].start)) 
                {
                    min_proc = 0; min_offset = index_cpu;
                    next_task = &cpu_schedule[index_cpu++];
                } 
                else if (index_gpu < gpu_schedule.size()) 
                {
                    min_proc = 1; min_offset = index_gpu;
                    next_task = &gpu_schedule[index_gpu++];
                }

                if (next_task) 
                {
                    const uint offset = min_offset;
                    const auto& schedule = *next_task;
                    const uint tid = schedule.task_id; auto& task = list_task[tid];

                    // std::cout << "Processing task: "; next_task->print();
                    // std::cout << "Performing additional computation for task ID: " << task_id << std::endl;

                    if (is_constraint_task[tid])
                    {
                        uint selected_buffer_idx = -1u;

                        if (!list_in[tid].empty())
                        {
                            for (const uint in_tid : list_in[tid])
                            {
                                auto& in_task = list_task[in_tid];
                                const uint read_in_idx = task_buffers[in_tid];
                                selected_buffer_idx = read_in_idx;
                                
                                task.buffer_ins.push_back(read_in_idx);
                                task.task_ins.push_back(in_tid);
                                
                                in_task.buffer_outs.push_back(read_in_idx);
                                in_task.task_outs.push_back(tid);

                                // task.buffer_in = read_in_idx;
                                // task.task_in = in_tid;
                                // in_task.buffer_out = read_in_idx;
                                // in_task.task_out = tid;

                                const uint left_tid = left_constraint_tid[min_proc];
                                if (left_tid != -1u)
                                {
                                    const auto& left_scheule = task_schedules[left_tid];

                                #if USE_MAIN_DEVICE
                                    if (list_out[left_tid].empty() || min_proc == 1) // GPU 默认加权，CPU 默认拷贝
                                #else
                                    if (true) // 全都参与加权
                                #endif
                                    {
                                        const uint left_buffer_idx = task_buffers[left_scheule.task_id];
                                        task.buffer_left = left_buffer_idx; 
                                        task.task_left = left_tid;
                                        // (in != -1) && (left != -1) : Weight
                                        // (in != -1)                 : Copy Read-In
                                        //               (left != -1) : Copy Left
                                    }
                                }
                            }
                            // else 
                            // {
                            //     task.buffer_left = 0; 
                            // }
                        }
                        else 
                        {
                            const uint left_task_idx = left_constraint_tid[min_proc];
                            if (left_task_idx != -1u && list_out[left_task_idx].empty())
                            {
                                // Directly get from left (no copy/weight)
                                selected_buffer_idx = task_buffers[left_task_idx];
                            }
                            else 
                            {
                                // Neet to copy from left, and there is no buffer that we can use
                                // So we need to allocate a new buffer
                                // TODO: Buffer pool
                                selected_buffer_idx = fn_update_buffer_idx();
                                if (left_task_idx != -1u) 
                                {   
                                    // Copy from left and 
                                    task.buffer_left = task_buffers[left_task_idx];
                                    task.task_left = left_task_idx;
                                }
                                else 
                                {
                                    // First iterative task in current processor
                                    task.buffer_left = input_buffer_mask;
                                    task.task_left = predict_position_tid;
                                }
                            }
                        }

                        task_buffers[tid] = selected_buffer_idx;
                        task.buffer_idx = selected_buffer_idx;
                        left_constraint_tid[min_proc] = tid;
                        // fast_format(" Task {:2}: left = {}, input = {}, buffer = {}  ({})", 
                        //     tid, 
                        //     task.buffer_left != -1u ? std::to_string(task.buffer_left) : "/", 
                        //     task.buffer_in != -1u ? std::to_string(task.buffer_in) : "/", 
                        //     selected_buffer_idx, taskNames.at(task.func_id));
                    }
                }
            }
        }
    }


    set_constant_computation_time_task();
    

    // Check
    for (uint proc = 0; proc < num_procs; proc++)
    {
        const auto& proc_tasks = proc_schedules[proc];
        if (proc_tasks.size() > 1)
        {
            for (uint i = 0; i < proc_tasks.size() - 1; i++)
            {
                const ScheduleEvent& curr_schedules = proc_tasks[i];
                const ScheduleEvent& next_schedules = proc_tasks[i + 1];
                if ((curr_schedules.end - curr_schedules.start != 0) && curr_schedules.start >= next_schedules.start)
                {
                    fast_print_err("Next Task Is Earliear Than Self : ", curr_schedules.get_str() + next_schedules.get_str());
                    fast_print_err(std::format("    {} and {}", 
                        taskNames.at(list_task[curr_schedules.task_id].func_id), 
                        taskNames.at(list_task[next_schedules.task_id].func_id)));
                }
            }
        }
    }
    
}
void Scheduler::make_wait_events()
{
    uint num_procs = communication_startup.size();
    // scheduler_shared_event = get_device()->newSharedEvent();

    if (num_procs != 2){
        fast_print_err("Sorry for the unsuported number of processors, \
            we only considering the transport between CPU and GPU");
        return;
        // num_procs = 2;
    }
    
    const uint num_tasks = list_task.size();

    ///
    /// Given the Scheduled List, How to Launch These Task on CPU/GPU
    ///    , and Handling the Consistent Relationship
    ///
    struct ProcBelongTo{
        uint proc;
        uint offset;
        ProcBelongTo() : proc(0), offset(0) {}
        ProcBelongTo(const uint& p, const uint& o) : proc(p), offset(o) {}
    };
    std::vector< ProcBelongTo > proc_belongs(num_tasks);

    for (uint proc = 0; proc < num_procs; proc++) {
        const ListSchedule& proc_schedule = proc_schedules[proc];
        for (uint i = 0; i < proc_schedule.size(); i++) {
            const ScheduleEvent& job = proc_schedule[i];
            proc_belongs[job.task_id] = ProcBelongTo(proc, i);
        }
    }

    std::vector< std::vector<uint> > list_signal(num_procs);
    std::vector< std::vector<uint> > list_wait(num_procs);
    for (uint proc = 0; proc < num_procs; proc++) 
    { 
        const uint num_tasks_allocated_to_proc = proc_schedules[proc].size();
        list_wait[proc].resize(num_tasks_allocated_to_proc, -1u);
        list_signal[proc].resize(num_tasks_allocated_to_proc, -1u);
    }

    // fast_format("CPU Size = {} , GPU Size = {}", proc_schedules[0].size(), proc_schedules[1].size());

    const bool print_events = false;

    ///
    /// 区分任务间的依赖 : 任务需要等待/signal
    ///
    if (print_events) 
        fast_print("\n------ Original Waiting & Signal Events ------\n");

    for (uint proc = 0; proc < num_procs; proc++){

        if (print_events) fast_print("Proc", proc);

        const ListSchedule& proc_schedule = proc_schedules[proc];
        std::vector<uint>& to_wait = list_wait[proc];

        std::vector< uint > list_waited_offsets;

        for (uint offset = 0; offset < proc_schedule.size(); offset++) {

            const ScheduleEvent& job = proc_schedule[offset];
            uint tid = job.task_id;
            auto& task = list_task[tid];
            const ListTask& preds = task.predecessors;
            /// Find the Last Node to Make Signal for Self
            uint pred_offset = -1u;
            float last_pred_time = 0.f;
            for (const auto& pred: preds) {
                bool in_anot_queue = proc_belongs[pred].proc != proc;
                if (in_anot_queue){
                    const ScheduleEvent& pred_job = task_schedules[pred];
                    if (pred_job.end > last_pred_time){
                        
                        last_pred_time = pred_job.end;
                        pred_offset = proc_belongs[pred].offset;
                        if (pred_job.end > job.start){
                            fast_print_err(std::format("Scheduling Result Miss Relying Relationship at tid {} -> {} ( Job Name : {} -> {}, Time : [{} -> {}] and [{} -> {}] )", 
                                pred, tid, 
                                taskNames.at(list_task[pred_job.task_id].func_id), taskNames.at(task.func_id), 
                                pred_job.start, pred_job.end, job.start, job.end));               
                        }
                    }
                }
            }
            if (pred_offset != -1u) {
                /// 一个task的signal可能被多个任务使用, 则按照最小的即可
                /// 到时候画图的时候可以把多种依赖关系都贴上去, 一对多, 多对一, cpu/gpu,gpu/cpu
                std::vector<uint>& to_signal = list_signal[1 - proc];
                uint target_signal = to_signal[pred_offset];
                if (target_signal == -1) { /// 如果没有被signal过
                    bool is_redundant_wait = false;
                    for (const auto& waited_offset : list_waited_offsets){
                        if (waited_offset >= pred_offset){
                            if (print_events) {
                                std::cout << "Drop Redundant Wait : " << offset << " Wait for " << pred_offset 
                                    << " , Already Exists : " << waited_offset << std::endl;
                            }
                            // uint pred = proc_schedules[1 - proc][pred_offset].task_id;
                            // uint waited_pred = proc_schedules[1 - proc][waited_offset].task_id;
                            // std::cout << "Drop Redundant Wait : " << i << " (" << Launcher::taskNames.at(task.func_id) 
                            //           << ") Wait for " << pred_offset  << " (" << Launcher::taskNames.at(list_task[pred].func_id) 
                            //           << ") , Already Exists : " << waited_offset << "(" << Launcher::taskNames.at(list_task[waited_pred].func_id) << ") " << std::endl;
                            is_redundant_wait = true;
                            break;
                        }
                    }
                    if (!is_redundant_wait) {
                        to_wait[offset] = pred_offset; 
                        to_signal[pred_offset] = offset; /// 它要signal我

                        // if(tid == 51){
                        //     fast_print("-----Find CheckHealthy-----", pred_offset, last_pred_time);
                        //     fast_print(i, "is to wait", pred_offset);
                        //     fast_print(pred_offset, "is to to_signal", i);
                        //     fast_print("\n\n");
                        // }

                        list_waited_offsets.push_back(pred_offset);

                        uint pred = proc_schedules[1 - proc][pred_offset].task_id;
                        const auto& pred_task = list_task[pred];

                        // std::cout << "   " << i << " Wait for "  << pred_offset << " , Task " << Launcher::taskNames.at(task.func_id) 
                        //     << " Rely on " << Launcher::taskNames.at(pred_task.func_id) << std::endl;
                        if(print_events) 
                            std::cout << "   " << offset << " Wait for "  << pred_offset << std::endl;
                    }

                }
                /// 没有else: 之前就等待过该task了 -> 合并等待
                // else{
                //     target_signal = std::min(target_signal, i);
                // }
            }
        }     
    }

    /// Proc : 0
    ///    0 Wait for 2
    ///    1 Wait for 4
    ///    2 Wait for 8
    ///    3 Wait for 7
    /// Proc : 1
    ///    9 Wait for 0
    ///    11 Wait for 1
    ///    17 Wait for 4

    
    ///
    /// Compute commit index (segment each queue into several commits)
    ///

    list_cmd_idx.resize(num_procs);
    launch_events.resize(num_procs);

    
    if(print_events)
        fast_print("\n------ Making Events From Waiting&Signal Flags ------\n");
    for (uint proc = 0; proc < num_procs; proc++) {

        if(print_events)
            fast_print("For Processor", proc);

        ListSchedule& schedules = proc_schedules[proc];
        const uint num_tasks_in_proc = schedules.size();

        const auto& waits = list_wait[proc];
        const auto& signals = list_signal[proc];
        std::vector<LaunchEvent>& events = launch_events[proc];

        std::vector<uint>& cmd_idxs = list_cmd_idx[proc];
        cmd_idxs.resize(num_tasks_in_proc);

        uint cmd_idx = 0; /// there're several 'sending commit buffer' operation
        uint start_idx = 0, end_idx = 0, wait_idx = -1u;
        for (uint i = 0; i < num_tasks_in_proc; i++){
            
            /// gpu : signal : set signal & commit & start new list
            ///       wait   : (commit & ) start new list & wait shared event in the end of the queue
            /// cpu : signal : set signal
            ///       wait   : wait target cmd buffer (cmd buffer idx)

            uint wait = waits[i];
            uint signal = signals[i];

            

            if(wait != -1u){
                
                if((end_idx != i - 1) && (i != 0)){
                    /// 上一个任务并不是上一个commit的结尾 -> 把上一个队列commit掉
                    end_idx = i - 1;

                    // std::cout << "   From Wait : " << start_idx << " to " << end_idx
                    //   << " wait / signal  = " << wait_idx << " / " << -1u << std::endl; 
                    
                    // if(i == 11 && proc == 0) { fast_print("id_check_healthy wait for", wait, wait_idx); }

                    events.push_back(LaunchEvent(start_idx, end_idx, wait_idx, -1u));
                    cmd_idx++;
                } 

                /// Drop Redundant Waiting Events
                // bool exist_waiting_task_before = false;
                // for(const auto& prev_event : events){
                //     if(prev_event.wait != 0 && prev_event.wait >= wait){
                //         exist_waiting_task_before = true;
                //         break;
                //     }
                // }
                // if(!exist_waiting_task_before){
                // }     
                // else{
                // }  
                
                wait_idx = wait;
                start_idx = i;
            }

            cmd_idxs[i] = cmd_idx; /// After Commit Last CommandBuffer

            if(signal != -1u || i == num_tasks_in_proc - 1){
                end_idx = i;
                uint signal_idx = signal; /// 谁在等我 : 临时有,临时建

                // std::cout << "   From Signal : " << start_idx << " to " << end_idx
                //       << " wait / signal  = " << wait_idx << " / " << signal_idx << std::endl; 
                
                /// If signal != -1u && i == num_tasks_in_proc - 1
                ///    => signal_idx = -1u, Still Does NOT Make Signal Event
                events.push_back(LaunchEvent(start_idx, end_idx, wait_idx, signal_idx));
                cmd_idx++;

                wait_idx = -1u; /// reset wait
                start_idx = i + 1;
            }

        }

        /// for GPU
        // if(proc == 1){
        //     list_cmd_buffer.resize(events.size() + 1);
        // }

    }

    // list_cmd_buffer.resize(20);
    // 
    ///
    /// Change the signal & it's corresponding wait into commit idx
    ///

    // fast_print("\n------ CommitIdx of Each Task ------\n");
    // for (uint proc = 0; proc < num_procs; proc++) {
    //     fast_print("Proc", proc);
    //     const auto& proc_cmd_idx = list_cmd_idx[proc];
    //     for (uint i = 0; i < proc_cmd_idx.size(); i++) {
    //         const auto& cmd_idx = proc_cmd_idx[i];
    //         std::cout << "   " << i << " : " << cmd_idx << std::endl;
    //     }
    // }

    if (print_events)
        fast_print("\n------ Change the Signal to CommitIdx ------\n");
    for (uint curr_proc = 0; curr_proc < num_procs; curr_proc++) {
        if (print_events) 
            fast_print("Proc", curr_proc);
        std::vector<LaunchEvent>& curr_proc_events = launch_events[curr_proc];
        std::vector<LaunchEvent>& anot_proc_events = launch_events[1 - curr_proc];
        for (uint cmd_idx = 0; cmd_idx < curr_proc_events.size(); cmd_idx++){
            auto& this_event = curr_proc_events[cmd_idx];
            
            uint target_offset = this_event.signal; /// offset
            if (target_offset != -1u){
                uint target_cmd_idx = list_cmd_idx[1 - curr_proc][target_offset];
                curr_proc_events[cmd_idx].signal = cmd_idx;
                anot_proc_events[target_cmd_idx].wait = cmd_idx;
                if(print_events){
                    std::cout << "  Change the Signal " << target_offset << " to CommitIdx " << cmd_idx 
                        << " , Target CommitIdx = " << target_cmd_idx  << std::endl;
                }
            }
        }
    }
  

    //
    // Costs
    //

    // double cost_cpu = 0.0, cost_gpu = 0.f;
    // for (const std::vector<float>& pair : computation_matrix)  { cost_cpu += pair[0]; cost_gpu += pair[1]; }
    // fast_format("Homogeneous Environment Cost in CPU = {:6.3f} , GPU = {:6.3f}", cost_cpu, cost_gpu);


    // double total_cpu = 0.0; double total_gpu = 0.0;
    // for (uint cmd_idx = 0; cmd_idx < launch_events[0].size(); cmd_idx++) {
    //     const LaunchEvent& event = launch_events[0][cmd_idx];
    //     for (uint i = event.start_idx; i <= event.end_idx; i++) {
    //         auto& cpu_jobs = proc_schedules[0][i];
    //         uint tid = cpu_jobs.task_id;
    //         total_cpu += computation_matrix[tid][0];
    //     }   
    // }
    // for (uint cmd_idx = 0; cmd_idx < launch_events[1].size(); cmd_idx++) {
    //     const LaunchEvent& event = launch_events[1][cmd_idx];
    //     for (uint i = event.start_idx; i <= event.end_idx; i++) {
    //         auto& cpu_jobs = proc_schedules[1][i];
    //         uint tid = cpu_jobs.task_id;
    //         total_gpu += computation_matrix[tid][1];
    //     }   
    // }

    // float cpu_back = proc_schedules[0].empty() ? 0.f : proc_schedules[0].back().end;
    // float gpu_back = proc_schedules[1].empty() ? 0.f : proc_schedules[1].back().end;
    // fast_format("Scheduling Result : CPU = {:6.3f} , GPU = {:6.3f} (Each Device Actually Working Time : CPU = {:6.3f} , GPU = {:6.3f}", 
    //     cpu_back, gpu_back, total_cpu, total_gpu);

    /// Check : All of the Waiting Events Has Target Signal 
    if (false) {
        for (uint proc = 0; proc < num_procs; proc++) {
            std::vector<LaunchEvent>& this_proc_events = launch_events[proc];
            std::vector<LaunchEvent>& anot_proc_events = launch_events[1 - proc];
            for(uint cmd_idx = 0; cmd_idx < this_proc_events.size(); cmd_idx++){
                auto& this_event = this_proc_events[cmd_idx];
        
                if(this_proc_events[cmd_idx].wait != -1u){
                    uint target_cmd_idx = this_proc_events[cmd_idx].wait;
                    // if(target_cmd_idx != -1u){
                        uint anot_signal = anot_proc_events[target_cmd_idx].signal;
                        if(target_cmd_idx != anot_signal){
                            fast_print_err("Not corresponding Signal and Waits:");
                            std::cerr << "    signal from : " << proc << " 's " << target_cmd_idx << " to proc " << 1 - proc << " 's " << anot_signal << "\n";
                        }
                    // }  
                }
                    
            }
        }
    }
    
    // print_proc_schedule();

    {
        /// 如何考虑gpu内部的并行?
    }
}



/// 
/// Print Information
///
void Scheduler::print_costs(bool use_sort){

    fast_print("Cost of each task (cpu & gpu) ");
    // for (auto tid : list_order) {
    //     auto& task = list_task[tid];
    //     std::cout << Launcher::taskNames.at(task.func_id) << " : " << task.list_cost[0] << " and " << task.list_cost[1] << std::endl;
    // }

    // Compute Rate
    std::vector< std::pair<uint, float> > list_rate(list_task.size());
    for(uint tid = 0; tid < list_task.size(); tid++){
        auto& task = list_task[tid];
        auto& costs = computation_matrix[tid];
        float rate = costs[0] / costs[1];
        list_rate[tid] = std::make_pair(tid, rate);
    }

    std::sort(list_rate.begin(), list_rate.end(), [&](const std::pair<uint, float>& left, const std::pair<uint, float>& right) -> bool{
        return left.second < right.second;
    });

    for(auto& pair : list_rate){
        const uint tid = pair.first;
        auto& costs = computation_matrix[tid];
        auto& task = list_task[tid];
        std::cout.width(10);
        std::cout << costs[0] << " and " << costs[1] << " in " << Launcher::taskNames.at(task.func_id) << std::endl;

        // float rate = pair.second;
        // fast_print(Launcher::taskNames.at(task.func_id), rate);
        // if(rate > 5.f)
        // {
        //     std::cout << "   cpu : ";
        //     for (auto dt : cost_list_cpu[tid]) {
        //         fast_print_single(dt);
        //     }
        //     std::cout << "\n   gpu : ";
        //     for (auto dt : cost_list_gpu[tid]) {
        //         fast_print_single(dt);
        //     }
        //     fast_print("");
        // }

    }
}
void Scheduler::print_schedule() {
    const uint num_procs = proc_schedules.size();
    const uint num_tasks = list_task.size();;
    std::vector<double> sum_t(2, 0.0);
    for (uint tid = 0; tid < num_tasks; tid++) {
        auto& task = list_task[tid];
        const auto& comp_mat = computation_matrix[tid];
        float cost_cpu = comp_mat[0];
        float cost_gpu = comp_mat[1];
        sum_t[0] += cost_cpu;
        if(cost_gpu != inf) sum_t[1] += cost_gpu;
        else                sum_t[0] += cost_cpu;
    }
    fast_print_single("Total Costs : "); fast_print("CPU", sum_t[0], "GPU", sum_t[1]);
    fast_print("Task of each processor (Copy me into python script) :");
    for (uint proc = 0; proc < num_procs; proc++) {
        fast_print("Processor", proc);
        const auto& proc_tasks = proc_schedules[proc];
        for (uint i = 0; i < proc_tasks.size(); i++) {
            const ScheduleEvent& task_scheduled = proc_tasks[i];
            uint tid = task_scheduled.task_id;
            const auto& task = list_task[tid];
            std::cout << "    " << i << " : " << tid << " , "  << task_scheduled.start << " to " << task_scheduled.end << " for " << Launcher::taskNames.at(task.func_id) << std::endl;
        }
    }
}
void Scheduler::print_scheduling_with_waiting_events(){

    const uint num_procs = proc_schedules.size();
    { 
        fast_print("\nFinal Launch Events");
        for (uint proc = 0; proc < num_procs; proc++) {
            std::cout << "Proc " << proc << " : \n" ;
            const std::vector<LaunchEvent>& this_proc_events = launch_events[proc];
            const std::vector<LaunchEvent>& anot_proc_events = launch_events[1-proc]; 
            const std::vector<ScheduleEvent>& this_proc_tasks = proc_schedules[proc];
            const std::vector<ScheduleEvent>& anot_proc_tasks = proc_schedules[1-proc];

            for(uint cmd_idx = 0; cmd_idx < this_proc_events.size(); cmd_idx++){
                auto& event = this_proc_events[cmd_idx];
                std::cout << "    " << cmd_idx << " : " << event.start_idx << " to " << event.end_idx;   

                std::string str_wait   = (event.wait == -1u)   ? "null" : std::to_string(event.wait);
                std::string str_signal = (event.signal == -1u) ? "null" : std::to_string(event.signal);
                std::cout << " , " << " Wait for " << str_wait  << " , Signal With " << str_signal << std::endl;

                {
                    for(uint i = event.start_idx; i <= event.end_idx; i++){ 
                        uint tid = this_proc_tasks[i].task_id;
                        if (list_task[tid].cluster_idx == 0)
                        {
                            std::cout << "      " << tid << " / " << Launcher::taskNames.at(list_task[tid].func_id) << std::endl;
                        }
                        else 
                        {
                            std::cout << "      " << tid << " / " << Launcher::taskNames.at(list_task[tid].func_id) << " : " << list_task[tid].cluster_idx << std::endl;
                        }
                    }   
                }
                
            }
        }
    }
    for (uint proc = 0; proc < num_procs; proc++) {
        double sum = 0;
        for(auto& costs : orig_computation_matrix){
            if(costs[proc] != inf){ sum += costs[proc]; }
            else sum += costs[0];
        }
        const auto& list = proc_schedules[proc];
        if(list.empty()) std::cout << "    Proc " << proc << " : " << sum << " : from " << 0 << " to " << 0 << std::endl;
        else             {
            float end_t = list.back().task_id == terminal_node ? list.size() > 1 ? list[list.size() - 2].end : 0 : list.back().end;
            std::cout << "    Proc " << proc << " : " << sum << " : from " << list[0].start << " to " << end_t << std::endl;}
    }
    
}
void Scheduler::print_proc_schedules(){
    fast_print("Task of each processor");
    const uint num_procs = proc_schedules.size();
    for (uint proc = 0; proc < num_procs; proc++) {
        const auto& proc_tasks = proc_schedules[proc];
        std::cout << "[";
        for (uint i = 0; i < proc_tasks.size(); i++) {
            const ScheduleEvent& task_scheduled = proc_tasks[i];
            uint tid = task_scheduled.task_id;
            auto& task = list_task[tid];
            std::cout << "(" << tid << ", "  << proc << ", " << task_scheduled.start << ", " << task_scheduled.end << ")";
            if(i != proc_tasks.size() - 1){
                std::cout << ", ";
            }
        }
        std::cout << "],\n";
    }
    print_tasks();
}
void Scheduler::print_schedule_to_graph_xpbd()
{

    if (proc_schedules.empty()) 
    {
        fast_print_err("Does Not Make Schedule, So We Can Not Print...");
        return;
    }

#define REMAP_TO_ARTICLE true

    std::set<std::pair<uint, uint>> task_ids;
    for (const auto& task_schedule : task_schedules) 
    { 
        const uint task_idx = task_schedule.task_id;  
        const auto& task = list_task[task_idx];
    #if REMAP_TO_ARTICLE
        task_ids.insert(std::make_pair(task.func_id, 0)); 
    #else
        task_ids.insert(std::make_pair(task.func_id, task.cluster_idx)); 
    #endif
    }

    std::map<std::pair<uint, uint>, uint> funcID_to_color_map;
    for (const auto& task_id : task_ids) {
        auto it = task_ids.find(task_id);
        if (it != task_ids.end()) {
            uint position = std::distance(task_ids.begin(), it);
            funcID_to_color_map.insert(std::make_pair(task_id, position));
        }
    }

    std::map<FunctionID, uint> funcid_color_map = {
        // {id_additional_root, -1u},
        // {id_additional_terminal, -1u},
        // {id_xpbd_constraint_last_node, -1u},
        // {id_xpbd_reset_collision_constrains, -1u},
        // {id_additional_terminal, -1u},
        // {id_xpbd_update_velocity, -1u},
        // {id_xpbd_copy_to_cpu_gpu, -1u},
        // {id_prepare_collision_detection_position, -1u},
        // {id_self_collision_scan_and_fill_in_vv, -1u},
        // {id_obstacle_collision_scan_and_fill_in_vf, -1u},
        // {id_graph_coloring_vivace_vv_wait_for_num_color, -1u},
        // {id_additional_terminal, -1u},

        {id_xpbd_predict_position, 1},
        {id_self_collision_update_vert_aabb_dcd, 2},
        {id_obstacle_collision_update_face_aabb_dcd, 3},
        {id_self_collision_apply_leaves_aabb, 4},
        {id_obstacle_collision_apply_leaves_aabb, 5},
        {id_self_collision_query_from_cloth_vert, 6},
        {id_obstacle_collision_query_from_cloth_vert, 7},
        {id_self_collision_reset_collision_system_and_lbvh, 8},
        {id_obstacle_collision_reset_collision_system_and_lbvh, 9},
        {id_self_collision_narrow_phase_vv, 10},
        {id_obstacle_collision_narrow_phase_vf, 11},
        {id_graph_coloring_vivace_vv_cloth, 12},
        {id_graph_coloring_vivace_vv_tet, 12},
        {id_xpbd_reset_constrains, 13},

        {id_vbd_all_in_one, 14},
        {id_sod_all_in_one, 14},
        {id_xpbd_constraint_stress_half, 14},

        {id_xpbd_constraint_stretch_mass_spring_half, 14},
        {id_xpbd_constraint_bending_half, 15},
        {id_xpbd_constraint_self_collision_vv_half_cloth, 16},
        {id_xpbd_constraint_obstacle_collision_vv_cloth, 17},
        // {id_xpbd_constraint_ground_collision_tet, 14},
        {id_xpbd_constraint_obstacle_collision_vv_tet, 22},
        {id_xpbd_constraint_self_collision_vv_half_tet, 23},
    };

    fast_print("\n Task of each processor");
    const uint num_procs = 2;
    for (uint proc = 0; proc < num_procs; proc++) {
        const auto& proc_tasks = proc_schedules[proc];

        if (proc_tasks.size() > 1)
        {
            for (uint i = 0; i < proc_tasks.size() - 1; i++)
            {
                const ScheduleEvent& curr_schedules = proc_tasks[i];
                const ScheduleEvent& next_schedules = proc_tasks[i + 1];
                if ((curr_schedules.end - curr_schedules.start != 0) && curr_schedules.start >= next_schedules.start)
                {
                    fast_print_err("Next Task Is Earliear Than Self : ", curr_schedules.get_str() + next_schedules.get_str());
                    fast_print_err(std::format("    {} and {}", 
                        taskNames.at(list_task[curr_schedules.task_id].func_id), 
                        taskNames.at(list_task[next_schedules.task_id].func_id)));
                }
            }
        }
        
        std::cout << "    [";
        for (uint i = 0; i < proc_tasks.size(); i++) {
            const ScheduleEvent& task_scheduled = proc_tasks[i];
            uint tid = task_scheduled.task_id;
            auto& task = list_task[tid];

        #if REMAP_TO_ARTICLE
            uint reflection_color; 
            if (funcid_color_map.find(task.func_id) != funcid_color_map.end())
            {
                reflection_color = funcid_color_map.at(task.func_id) + task.cluster_idx;
            }
            else 
            {
                reflection_color = 0;
            }
            // if (reflection_color == -1u) reflection_color = 0;
            // if (reflection_color == -1u) continue;
        #else
            auto orig_color = std::make_pair(task.func_id, task.cluster_idx);
            uint reflection_color = funcID_to_color_map[orig_color];
        #endif

            // std::cout << std::format("({}, {}, {:4.3f}, {:4.3f})", reflection_color, proc, task_scheduled.start, task_scheduled.end);
            std::cout << std::format("({}, {}, {}, {:4.3f}, {:4.3f})", reflection_color, tid, task.buffer_idx, task_scheduled.start, task_scheduled.end);
            if (abs(task_scheduled.end - task_scheduled.start - computation_matrix[task_scheduled.task_id][task_scheduled.proc]) > 0.001)
            {
                fast_print_err("   Scheduling Result Does Not Match The Computation Matrix : ", 
                    std::format(" from {} to {} , should be {}", task_scheduled.start, task_scheduled.end, computation_matrix[task_scheduled.task_id][task_scheduled.proc]));
            }
            if (i != proc_tasks.size() - 1){
                std::cout << ", ";
            }
        }
        std::cout << "],\n";
    }

    

    // Cross Device Connection
    std::cout << "    [";
    for (const ScheduleEvent& curr_schedule : task_schedules) {
        uint tid = curr_schedule.task_id;
        auto& task = list_task[tid];

        if (!task.task_ins.empty()) 
        {
            for (auto input_task : task.task_ins)
            {
                const ScheduleEvent& in_schedule = task_schedules[input_task];
                std::cout << std::format("({}, {}, {}, {}, {:4.3f}, {:4.3f}), ", in_schedule.proc, input_task, curr_schedule.proc, tid, in_schedule.end, curr_schedule.start);
            }
        }   
        // if (task.task_in != -1u) 
        // {
        //     const ScheduleEvent& in_schedule = task_schedules[task.task_in];
        //     std::cout << std::format("({}, {}, {}, {}, {:4.3f}, {:4.3f}), ", in_schedule.proc, task.task_in, curr_schedule.proc, tid, in_schedule.end, curr_schedule.start);
        // }   
    }
    std::cout << "],\n";

    // Inner Device Copy
    std::cout << "    [";
    for (const ScheduleEvent& curr_schedule : task_schedules) {
        uint tid = curr_schedule.task_id;
        auto& task = list_task[tid];

        if (task.task_left != -1u) 
        {
            const ScheduleEvent& in_schedule = task_schedules[task.task_left];
            std::cout << std::format("({}, {}, {}, {}, {:4.3f}, {:4.3f}), ", in_schedule.proc, task.task_left, curr_schedule.proc, tid, in_schedule.end, curr_schedule.start);
        }   
    }
    std::cout << "],\n";


    
    // for (uint proc = 0; proc < num_procs; proc++) {
    //     const auto& proc_tasks = proc_schedules[proc];

    //     if (proc_tasks.size() > 1)
    //     {
    //         fast_print("\n Task of each processor", proc);
    //         for (uint i = 0; i < proc_tasks.size(); i++) {
    //             const ScheduleEvent& task_scheduled = proc_tasks[i];
    //             uint tid = task_scheduled.task_id;
    //             auto& task = list_task[tid];

    //             auto orig_color = std::make_pair(task.func_id, task.cluster_idx);
    //             uint reflection_color = funcID_to_color_map[orig_color];

    //             fast_format("({}, {}, {:4.3f}, {:4.3f})", reflection_color, taskNames.at(task.func_id), task_scheduled.start, task_scheduled.end);
    //         }   
    //     }
    // }


    std::cout << "\nMeaning of Each Number (Color/Name) : ";
    for (const auto& pair : funcID_to_color_map) {
        const auto task = pair.first;
        const uint& color = pair.second;
        if (task.second != 0) {
            std::cout << std::format("({}, {}, {}) ", color, taskNames.at(FunctionID(task.first)), task.second);
        }
        else  {
            std::cout << std::format("({}, {}) ", color, taskNames.at(FunctionID(task.first)));
        }
    }
    fast_print();
    fast_print();
    
    // auto usages = get_proc_usage();
    // auto end_time = get_scheduled_end_time();
    // auto speedups = get_scheduled_speedup();
 
    // fast_print();
}
void Scheduler::print_tasks(){
    fast_print("List Tasks : ");
    for (uint tid = 0; tid < list_task.size(); tid++) {
        const auto& task = list_task[tid];
        task.print_with_cluster(tid);
    }
}
void Scheduler::print_dag() {
    fast_print("nodes = [");
    for (size_t i = 0; i < list_task.size(); ++i) 
    {
        fast_format("    ({}, {}),", i, int(list_task[i].func_id));
    }
    fast_print("]");
    fast_print("edges = [");
    for (size_t i = 0; i < list_task.size(); ++i) 
    {
        for (uint succ : list_task[i].successors) 
        {
            fast_format("    ({}, {}),", i, succ);
            std::cout << "    (" << i << ", " << succ << "),\n";
        }
    }
    fast_print("]");
}

float Scheduler::get_theoretical_time()
{
    const uint num_tasks = list_task.size();
    std::vector<ListCost> computation_matrix_copy(computation_matrix);
    std::sort(computation_matrix_copy.begin(), computation_matrix_copy.end(), [&](const ListCost& left, const ListCost& right)
    {
        // CPU / GPU => 越低，则越利好 CPU
        return left[0]/left[1] < right[0]/right[1];
    });
    
    std::vector<float> list_computation_matrix_cpu(num_tasks);
    std::vector<float> list_computation_matrix_gpu(num_tasks);
    for (uint i = 0; i < num_tasks; i++) 
    {
        list_computation_matrix_cpu[i] = computation_matrix_copy[i][0];
        list_computation_matrix_gpu[i] = computation_matrix_copy[i][1];
    }
    const float* ptr_cpu = list_computation_matrix_cpu.data();
    const float* ptr_gpu = list_computation_matrix_gpu.data();
    float curr_sum_cpu = 0;
    float curr_sum_gpu = 0;
    float curr_min_time = 0;
    for (uint i = 0; i < num_tasks; i++) 
    {
        curr_sum_cpu = std::accumulate(ptr_cpu, ptr_cpu + i, 0.f) + i * communication_cost_matrix_uma[0][0];
        curr_sum_gpu = std::accumulate(ptr_gpu + i + 1, ptr_gpu + list_computation_matrix_gpu.size(), 0.f) + (computation_matrix_copy.size() - i) * communication_cost_matrix_uma[1][1];
        if (curr_sum_cpu >= curr_sum_gpu) 
        {
            float prev_sum_gpu = std::accumulate(ptr_gpu + i, ptr_gpu + list_computation_matrix_gpu.size(), 0.f);
            curr_min_time = std::min(curr_sum_cpu, prev_sum_gpu);
            break;
        }
    }
    return curr_min_time;
}
float Scheduler::get_scheduled_end_time()
{
    float end_time = 0.0f; 
    for (uint proc = 0; proc < proc_schedules.size(); proc++)
    {
        end_time = std::max(proc_schedules[proc].empty() ? 0.f : proc_schedules[proc].back().end, end_time);
    }
    return end_time;
}
std::vector<float> Scheduler::get_scheduled_speedups()
{
    float end_time = get_scheduled_end_time();
    uint num_proc = proc_schedules.size();
    std::vector<float> speedups(num_proc, 0.f);
    for (uint proc = 0; proc < num_proc; proc++)
    {
        speedups[proc] = (summary_of_costs_each_device[proc] - end_time) / end_time;
    }
    return speedups;
}
std::vector<float> Scheduler::get_proc_usage()
{
    float end_time = get_scheduled_end_time();
    std::vector<float> proc_usage(proc_schedules.size(), 0.f);
    for (uint proc = 0; proc < proc_schedules.size(); proc++)
    {
        float curr_proc_allocated_task_sum = 0.0f;
        for (uint i = 0; i < proc_schedules[proc].size(); i++) 
        {
            auto& cpu_jobs = proc_schedules[proc][i];
            uint tid = cpu_jobs.task_id;
            curr_proc_allocated_task_sum += computation_matrix[tid][0] + communication_cost_matrix_uma[0][0];
        }
        proc_usage[proc] = curr_proc_allocated_task_sum / end_time;
        // fast_format("Usage in proc {} = {:4.2f}\%", curr_proc_allocated_task_sum / end_time * 100);
    }
    return proc_usage;
}
void Scheduler::print_speedups_to_each_device()
{
    float end_time = get_scheduled_end_time();
    uint num_proc = proc_schedules.size();
    std::vector<float> speedups(num_proc, 0.f);
    for (uint proc = 0; proc < num_proc; proc++)
    {
        float speedup = (summary_of_costs_each_device[proc] - end_time) / end_time;
        fast_format("Speedup to proc {} = {:4.2f}\% (From {:4.2f} to {:4.2f} ) ", proc, speedup * 100, summary_of_costs_each_device[proc], end_time);
    }
}


void Scheduler::reset_scheduler_system()
{
    list_task.clear();
    list_order.clear();
    sorted_nodes.clear();
    // list_virtual_sync.clear();
    orig_list_task.clear();
    orig_computation_matrix.clear(); 
    list_convergence.clear();
    list_additional_task_orig_tid.clear();
    task_schedules.clear();
    ranku.clear();
    proc_schedules.clear();
    list_cmd_idx.clear();
    launch_events.clear();
    computation_matrix.clear();

    list_task_merged.clear();
    task_to_merged_task_map.clear();
    task_schedules_merged.clear();
    proc_schedules_merged.clear();

    assemble_implementations.clear();
    constraint_task_orders.clear();
}
void Scheduler::test_case_2002()
{

    const uint num_procs = 3;
    const uint num_tasks = 10;
    const uint num_edges = 15;
    
    /// 10
    list_task.clear();
    list_task.resize(num_tasks);
    computation_matrix.clear();
    computation_matrix.resize(num_tasks);

    computation_matrix[0] = {14, 16, 9};
    computation_matrix[1] = {13, 19, 18};
    computation_matrix[2] = {11, 13, 19};
    computation_matrix[3] = {13, 8,  17};
    computation_matrix[4] = {12, 13, 10};
    computation_matrix[5] = {13, 16, 9};
    computation_matrix[6] = {7,  15, 11};
    computation_matrix[7] = {5,  11, 14};
    computation_matrix[8] = {18, 12, 20};
    computation_matrix[9] = {21, 7,  16};
    
    /// 15
    set_connect(0, 1, 18.f);
    set_connect(0, 2, 12.f);
    set_connect(0, 3, 9.f); 
    set_connect(0, 4, 11.f);
    set_connect(0, 5, 14.f);
    set_connect(1, 7, 19.f);
    set_connect(1, 8, 16.f);
    set_connect(2, 6, 23.f);
    set_connect(3, 7, 27.f);
    set_connect(3, 8, 23.f);
    set_connect(4, 8, 13.f);
    set_connect(5, 7, 15.f);
    set_connect(6, 9, 17.f);
    set_connect(7, 9, 11.f);
    set_connect(8, 9, 13.f);
    
    /// Scheduling
    topological_sort();

    standardizing_dag({
        [&](const Launcher::LaunchParam& param) {},
        [&](const Launcher::LaunchParam& param) {},
        [&](const Launcher::LaunchParam& param) {}
    });

    scheduler_dag();
 
}

void Scheduler::set_sync_count(const uint& input_sync_count) { sync_count = input_sync_count; }
void Scheduler::get_dag_from(const Scheduler& input_scheduler)
{
    /// Can NOT Use 'std::memcpy', Cause struct 'Task' Has 'std::vector'
    list_task.resize(input_scheduler.list_task.size());
    for (uint tid = 0; tid < input_scheduler.list_task.size(); tid++) {
        list_task[tid] = input_scheduler.list_task[tid];
    }
    assemble_implementations = input_scheduler.assemble_implementations;
    constraint_task_orders = input_scheduler.constraint_task_orders;
}
void Scheduler::set_constraint_task_orders(const std::vector<uint>& orders)
{
    constraint_task_orders = orders;
}
void Scheduler::set_constant_computation_time_task()
{
    for (auto& task : list_task)
    {
        if (
            task.func_id ==  id_xpbd_predict_position
            || task.func_id ==  id_xpbd_reset_constrains
            || task.func_id ==  id_xpbd_update_velocity
            || task.func_id ==  id_xpbd_copy_to_cpu_gpu
            //    task.func_id ==  id_xpbd_constraint_stretch_mass_spring_half
            // || task.func_id ==  id_xpbd_constraint_stretch_fem_half
            // || task.func_id ==  id_xpbd_constraint_stress_half
            // || task.func_id ==  id_xpbd_constraint_bending_half
            // || task.func_id ==  id_xpbd_constraint_ground_collision_cloth
            // || task.func_id ==  id_xpbd_constraint_ground_collision_tet
            || task.func_id ==  id_prepare_collision_detection_position
            || task.func_id ==  id_self_collision_reset_collision_system_and_lbvh
            || task.func_id ==  id_obstacle_collision_reset_collision_system_and_lbvh
            // || task.func_id ==  id_self_collision_update_vert_aabb_dcd
            // || task.func_id ==  id_obstacle_collision_update_face_aabb_dcd
            // || task.func_id ==  id_self_collision_apply_leaves_aabb
            // || task.func_id ==  id_obstacle_collision_update_face_aabb_dcd
            )
        {
            task.is_computation_time_constant = true;
        }
    }
}
void Scheduler::set_as_sync_as_possible(const std::vector<Task>& assemble_impl)
{
    // list_task.resize(assemble_impl.size());
    // for (uint tid = 0; tid < assemble_impl.size(); tid++) 
    // {
    //     assemble_implementations[tid] = assemble_impl[tid];
    // }
}

uint Scheduler::add_task(const Task& task)
{

    uint task_id = list_task.size();
    list_task.push_back(task);
    return task_id;

}
Task& Scheduler::get_task_by_tid(const uint& task_id)
{
    return list_task[task_id];
}
uint Scheduler::find_task_by_func_id(const Launcher::FunctionID id) 
{ 
    for (uint tid = 0; tid < list_task.size(); tid++) {
        const auto& task = list_task[tid];
        if(task.func_id == id) 
        {
            std::cout << "find funcid in " << tid << std::endl;
            return tid;
        }
    } 
    std::cerr << "Does not find Task " << id  << " : " << Launcher::taskNames.at(id) << std::endl;
    return -1u;
}

// TODO: stream like: task1 << task2 << task3 << task4
void Scheduler::set_connect(const uint& task_idx1, const uint& task_idx2, const float& weight)
{
    auto& task1 = list_task[task_idx1];
    auto& task2 = list_task[task_idx2];

    uint offset = task1.successors.size();
    task1.add_front(task_idx2);
    task1.list_weight.push_back(weight);
    
    task2.add_back(task_idx1);
    // task2.list_offset.push_back(offset);
}
float Scheduler::delete_connect(const uint& task_idx1, const uint& task_idx2)
{
    auto& task1 = list_task[task_idx1];
    auto& task2 = list_task[task_idx2];

    auto find_1_2 = std::find(task1.successors.begin(), task1.successors.end(), task_idx2);
    if (find_1_2 == task1.successors.end()) 
    {
        fast_print_err(std::format("Task 2 In Not The Successors of Task 1, Desire {} -> {}", 
            taskNames.at(task1.func_id), taskNames.at(task2.func_id)));
        return 0;
    }
    uint idx = std::distance(
        task1.successors.begin(), 
        find_1_2);
    task1.successors.erase(task1.successors.begin() + idx);
    float weight = *(task1.list_weight.begin() + idx);
    task1.list_weight.erase(task1.list_weight.begin() + idx);

    auto find_2_1 = std::find(task2.predecessors.begin(), task2.predecessors.end(), task_idx1);
    if (find_2_1 == task2.predecessors.end()) 
    {
        fast_print_err(std::format("Task 1 In Not The Predecessors of Task 2, Desire {} <- {}", 
            taskNames.at(task2.func_id), taskNames.at(task1.func_id)));
        return 0;
    }
    task2.predecessors.erase(find_2_1);
    return weight;
}

};