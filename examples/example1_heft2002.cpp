
#include <iostream>
#include <launcher.h>

int main()
{
    std::cout << "Hello, Heft!" << std::endl;
    
    Launcher::Scheduler scheduler;

    //
    // Register DAG
    //
    const uint num_procs = 3;
    const uint num_tasks = 10;
    const uint num_edges = 15;
    
    Launcher::Implementation ipm_xpbd_0(0, [&](const Launcher::LaunchParam& param){ fast_format(" Run task {} in processor 1", param.cluster_idx); });
    Launcher::Implementation imp_xpbd_1(1, [&](const Launcher::LaunchParam& param){ fast_format(" Run task {} in processor 2", param.cluster_idx); });
    Launcher::Implementation imp_xpbd_2(2, [&](const Launcher::LaunchParam& param){ fast_format(" Run task {} in processor 3", param.cluster_idx); });

    auto fn_add_task_template = [&scheduler, &ipm_xpbd_0, &imp_xpbd_1, &imp_xpbd_2](const uint id)
    {
        const uint task_idx = scheduler.add_task(Launcher::Task(
            Launcher::id_task_heft_2002, 
            id, // clusterIdx, you can replace the identifier as you want
            {ipm_xpbd_0, imp_xpbd_1, imp_xpbd_2}
        ));
        return task_idx; // The index in 'list_task'
    };
    
    const uint tid_1  = fn_add_task_template(1);
    const uint tid_2  = fn_add_task_template(2);
    const uint tid_3  = fn_add_task_template(3);
    const uint tid_4  = fn_add_task_template(4);
    const uint tid_5  = fn_add_task_template(5);
    const uint tid_6  = fn_add_task_template(6);
    const uint tid_7  = fn_add_task_template(7);
    const uint tid_8  = fn_add_task_template(8);
    const uint tid_9  = fn_add_task_template(9);
    const uint tid_10 = fn_add_task_template(10);
    

    //
    // Set connection (and communication matrix)
    //
    scheduler.set_connect(tid_1, tid_2,  18.f);
    scheduler.set_connect(tid_1, tid_3,  12.f);
    scheduler.set_connect(tid_1, tid_4,  9.f); 
    scheduler.set_connect(tid_1, tid_5,  11.f);
    scheduler.set_connect(tid_1, tid_6,  14.f);
    scheduler.set_connect(tid_2, tid_8,  19.f);
    scheduler.set_connect(tid_2, tid_9,  16.f);
    scheduler.set_connect(tid_3, tid_7,  23.f);
    scheduler.set_connect(tid_4, tid_8,  27.f);
    scheduler.set_connect(tid_4, tid_9,  23.f);
    scheduler.set_connect(tid_5, tid_9,  13.f);
    scheduler.set_connect(tid_6, tid_8,  15.f);
    scheduler.set_connect(tid_7, tid_10, 17.f);
    scheduler.set_connect(tid_8, tid_10, 11.f);
    scheduler.set_connect(tid_9, tid_10, 13.f);

    //
    // Set computation matrix
    //
    scheduler.computation_matrix.resize(num_tasks);
    scheduler.computation_matrix[tid_1] = {14, 16, 9};
    scheduler.computation_matrix[tid_2] = {13, 19, 18};
    scheduler.computation_matrix[tid_3] = {11, 13, 19};
    scheduler.computation_matrix[tid_4] = {13, 8,  17};
    scheduler.computation_matrix[tid_5] = {12, 13, 10};
    scheduler.computation_matrix[tid_6] = {13, 16, 9};
    scheduler.computation_matrix[tid_7] = {7,  15, 11};
    scheduler.computation_matrix[tid_8] = {5,  11, 14};
    scheduler.computation_matrix[tid_9] = {18, 12, 20};
    scheduler.computation_matrix[tid_10] = {21, 7,  16};

    //
    // Some constant
    //
    scheduler.communication_speed_matrix = {
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };
    
    scheduler.communication_cost_matrix_uma = {}; // If we use Uniform-Memory-Architecture, 
                                                  // then the data transfering cost is constant across all tasks 
                                                  // (Graphics API level waiting delays)
    scheduler.communication_startup = {0, 0, 0}; // First call cost

    //
    // Compute total cost of running all tasks on each device (For evaluating the accerate rate)
    // 'summary_of_costs_each_device' maybe be higher than the sum of each task due to inner-device cost.
    //          Espacially for the GPU (about 0.01m), then the diagonal of 'communication_speed_matrix' are not zero
    //
    scheduler.summary_of_costs_each_device = {0, 0, 0};
    for (uint proc = 0; proc < num_procs; proc++)
    {
        const float comm = scheduler.fn_get_inner_communication_cost(proc); // should be 0
        for (uint tid = 0; tid < num_tasks; tid++)
        {
            scheduler.summary_of_costs_each_device[proc] += scheduler.computation_matrix[tid][proc] + comm;
        }
    }

    //
    // Make scheduling
    //
    if (scheduler.topological_sort()) // Also, we will check whether the connection relationship (DAG) is correct
    {
        scheduler.print_sort_by_typology();

        scheduler.standardizing_dag(); // Add additional root and terminal node 

        scheduler.set_print_scheduling_datail(true);
        scheduler.scheduler_dag();
        
        scheduler.print_proc_schedules();

        scheduler.print_speedups_to_each_device();
        
        // scheduler.make_wait_events(); 

        // scheduler.print_schedule_to_graph_xpbd();
    }

    return 0;
}