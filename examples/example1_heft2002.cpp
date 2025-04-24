
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
    
    Launcher::Implementation ipm_xpbd_0(0, [&](const Launcher::LaunchParam& param){ fast_format(" Runtask {} in processor 1", param.cluster_idx); });
    Launcher::Implementation imp_xpbd_1(1, [&](const Launcher::LaunchParam& param){ fast_format(" Runtask {} in processor 2", param.cluster_idx); });
    Launcher::Implementation imp_xpbd_2(2, [&](const Launcher::LaunchParam& param){ fast_format(" Runtask {} in processor 3", param.cluster_idx); });

    for (uint tid = 0; tid < num_tasks; tid++)
    {
        scheduler.add_task(Launcher::Task(
            Launcher::id_task_heft_2002, 
            tid, // clusterIdx, you can replace the identifier as you want
            {ipm_xpbd_0, imp_xpbd_1, imp_xpbd_2}
        ));
    }

    //
    // Set computation matrix & communication matrix
    //
    scheduler.computation_matrix.resize(num_tasks);
    scheduler.computation_matrix[0] = {14, 16, 9};
    scheduler.computation_matrix[1] = {13, 19, 18};
    scheduler.computation_matrix[2] = {11, 13, 19};
    scheduler.computation_matrix[3] = {13, 8,  17};
    scheduler.computation_matrix[4] = {12, 13, 10};
    scheduler.computation_matrix[5] = {13, 16, 9};
    scheduler.computation_matrix[6] = {7,  15, 11};
    scheduler.computation_matrix[7] = {5,  11, 14};
    scheduler.computation_matrix[8] = {18, 12, 20};
    scheduler.computation_matrix[9] = {21, 7,  16};

    scheduler.set_connect(0, 1, 18.f);
    scheduler.set_connect(0, 2, 12.f);
    scheduler.set_connect(0, 3, 9.f); 
    scheduler.set_connect(0, 4, 11.f);
    scheduler.set_connect(0, 5, 14.f);
    scheduler.set_connect(1, 7, 19.f);
    scheduler.set_connect(1, 8, 16.f);
    scheduler.set_connect(2, 6, 23.f);
    scheduler.set_connect(3, 7, 27.f);
    scheduler.set_connect(3, 8, 23.f);
    scheduler.set_connect(4, 8, 13.f);
    scheduler.set_connect(5, 7, 15.f);
    scheduler.set_connect(6, 9, 17.f);
    scheduler.set_connect(7, 9, 11.f);
    scheduler.set_connect(8, 9, 13.f);

    scheduler.communication_speed_matrix = {
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };
    scheduler.communication_cost_matrix_uma = {};
    scheduler.communication_startup = {0, 0, 0};

    //
    // Compute sum of costs for each device
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
    // Schedule
    //
    if (scheduler.topological_sort())
    {
        scheduler.print_sort_by_typology();

        scheduler.standardizing_dag();

        scheduler.scheduler_dag();
        
        scheduler.print_proc_schedules();

        scheduler.print_speedups_to_each_device();
        
        // scheduler.make_wait_events(); 

        // scheduler.print_schedule_to_graph_xpbd();
    }
    

    return 0;
}