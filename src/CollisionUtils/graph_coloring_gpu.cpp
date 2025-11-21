#include "graph_coloring_gpu.h"
#include "command_list.h"
#include "gpu_algorism.h"
#include <set>

void RandomGraphColoringGPU::init_graph_coloring_system(VivaceColoringData& input_data, XpbdSelfCollision& input_self_collision, RandomGraphColoringCPU& input_vivace_cpu)
{
    vivace_data = &input_data;
    self_collision_data = &input_self_collision;
    vivace_cpu = &input_vivace_cpu;

#if __APPLE__

    NS::Error* err;

    std::string full_path_vivace_collision = std::string(SELF_RESOURCES_PATH) + std::string("/metal_libs/") + std::string("graph_coloring.metallib");
    MTL::Library* library_vivace = get_device() -> newLibrary(std_string_to_ns_string(full_path_vivace_collision), &err);
    check_err(library_vivace, err);

    fn_scan_uncolored_set.load(library_vivace, "scan_uncolored_set_GPU");
    fn_copy_scaned_indices_from.load(library_vivace, "copy_scaned_indices_from");
    fn_reduce_degree_vv.load(library_vivace, "reduce_degree_vv");
    fn_reduce_degree_vf.load(library_vivace, "reduce_degree_vf");
    fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.load(library_vivace, 
            "reduce_degree_second_pass_AND_set_max_color_from_global_degree");
    fn_init_palette.load(library_vivace, "init_palette");
    fn_tentative_coloring.load(library_vivace, "tentative_coloring");
    fn_copy_colred.load(library_vivace, "copy_colred");
    fn_conflict_resolution_vv.load(library_vivace, "conflict_resolution_per_element_vv");
    fn_conflict_resolution_vf.load(library_vivace, "conflict_resolution_per_element_vf");
    fn_update_palatte_from_current_tentative_coloring_result_vv.load(library_vivace, "update_palatte_from_current_tentative_coloring_result_vv");
    fn_feed_the_hungry.load(library_vivace, "feed_the_hungry");
    fn_put_rest_vertices_into_additional_color.load(library_vivace, "put_rest_vertices_into_additional_color");
    fn_put_rest_vertices_into_random_color.load(library_vivace, "put_rest_vertices_into_random_color");
    fn_scan_num_verts_in_color.load(library_vivace, "scan_num_verts_in_color");
    fn_fill_in_cluster_indices_vv.load(library_vivace, "fill_in_cluster_indices_vv");
    
#endif
}
//
// Graph Coloring
//
void RandomGraphColoringGPU::graph_coloring_vivace()
{    
    //
    // Initialization
    //
    reduce_degree_and_set_max_color_from_max_degree();
        
    scan_uncolored_set(0);

    init_palette();
    
    uint max_loop = VivaceGraphCloring::max_graph_coloring_colors; 

    for (uint curr_loop = 0; curr_loop < max_loop; curr_loop++) 
    {
        tentative_coloring(curr_loop);

        if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVV) {  conflict_resolution_per_element_vv(curr_loop); }
        if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVF) {  conflict_resolution_per_element_vf(curr_loop); }
        
        scan_uncolored_set(curr_loop + 1);

        feed_the_hungry(curr_loop + 1);  
    }

    // put_rest_vertices_into_random_color();

    make_cluster_indirect_cmd_buffer();
}

void RandomGraphColoringGPU::reduce_degree_and_set_max_color_from_max_degree()
{
    if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVV)
    {
        get_command_list().add_task(fn_reduce_degree_vv);
        fn_reduce_degree_vv.bind_ptr(self_collision_data->narrow_phase_list_indices_vv);
        fn_reduce_degree_vv.bind_ptr(self_collision_data->vert_VV_num_narrow_phase);
        fn_reduce_degree_vv.bind_ptr(vivace_data->uncolored_verts);
        fn_reduce_degree_vv.bind_ptr(vivace_data->num_verts_in_cluster);
        fn_reduce_degree_vv.bind_ptr(vivace_data->verts_prefix_in_cluster);
        fn_reduce_degree_vv.bind_ptr(vivace_data->colored);
        fn_reduce_degree_vv.bind_ptr(vivace_data->c_v);
        fn_reduce_degree_vv.bind_ptr(self_collision_data->collision_count);
        fn_reduce_degree_vv.bind_ptr(vivace_data->block_min_max_degree);
        fn_reduce_degree_vv.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0);

        get_command_list().add_task(fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->num_verts_in_cluster);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->uncolored_verts_indirect_cmd_buffer);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->num_colors_self_collision);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(self_collision_data->collision_count);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->block_min_max_degree);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.launch_async(SECOND_REDUCE_DIM, SECOND_REDUCE_DIM);
    }
    else if (vivace_data->element_type == VivaceGraphColoringElementTypePerVertVF)
    {
        get_command_list().add_task(fn_reduce_degree_vf);
        fn_reduce_degree_vf.bind_ptr(self_collision_data->narrow_phase_list_indices_vf);
        fn_reduce_degree_vf.bind_ptr(self_collision_data->vert_VV_num_narrow_phase);
        fn_reduce_degree_vf.bind_ptr(vivace_data->uncolored_verts);
        fn_reduce_degree_vf.bind_ptr(vivace_data->num_verts_in_cluster);
        fn_reduce_degree_vf.bind_ptr(vivace_data->verts_prefix_in_cluster);
        fn_reduce_degree_vf.bind_ptr(vivace_data->colored);
        fn_reduce_degree_vf.bind_ptr(vivace_data->c_v);
        fn_reduce_degree_vf.bind_ptr(self_collision_data->collision_count);
        fn_reduce_degree_vf.bind_ptr(vivace_data->block_min_max_degree);
        fn_reduce_degree_vf.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0);

        get_command_list().add_task(fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->num_verts_in_cluster);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->uncolored_verts_indirect_cmd_buffer);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->num_colors_self_collision);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(self_collision_data->collision_count);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.bind_ptr(vivace_data->block_min_max_degree);
        fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree.launch_async(SECOND_REDUCE_DIM, SECOND_REDUCE_DIM);
    }

}
void RandomGraphColoringGPU::scan_uncolored_set(const uint curr_loop)
{
    auto& uncolored_verts_copy = vivace_data->clusterd_constraint_self_collision;
    // const uint prev_uncolored = curr_loop == 0 ? self_collision_data->collision_count[0] : fn_get_current_uncolored_count(curr_loop - 1);

    auto& uncolored_verts_indirect_cmd_buffer = 
        curr_loop == 0 ? self_collision_data->self_collision_indirect_cmd_buffer :
        self_collision_data->self_collision_indirect_cmd_buffer;


    //
    // Since We Have Not Update The Indirect Command Buffer Yet, So We Use The Indirect Command Buffer In The Previous Loop
    //
    get_command_list().add_task(fn_scan_uncolored_set);
    fn_scan_uncolored_set.bind_ptr(self_collision_data->collision_count);
	fn_scan_uncolored_set.bind_ptr(vivace_data->uncolored_verts);
	fn_scan_uncolored_set.bind_ptr(uncolored_verts_copy);
	fn_scan_uncolored_set.bind_ptr(vivace_data->uncolored_verts_count);
	fn_scan_uncolored_set.bind_ptr(vivace_data->colored);
	fn_scan_uncolored_set.bind_constant(curr_loop);
    if (curr_loop == 0)
        fn_scan_uncolored_set.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0); 
    else 
        fn_launch_function_in_loop(curr_loop - 1, fn_scan_uncolored_set);
        // fn_scan_uncolored_set.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, curr_loop - 1);


    get_command_list().add_task(fn_copy_scaned_indices_from);
    fn_copy_scaned_indices_from.bind_ptr(self_collision_data->collision_count);
	fn_copy_scaned_indices_from.bind_ptr(uncolored_verts_copy);
    fn_copy_scaned_indices_from.bind_ptr(vivace_data->uncolored_verts);
	fn_copy_scaned_indices_from.bind_ptr(vivace_data->uncolored_verts_count);
	fn_copy_scaned_indices_from.bind_ptr(vivace_data->uncolored_verts_indirect_cmd_buffer);
	fn_copy_scaned_indices_from.bind_constant(curr_loop);
    if (curr_loop == 0)
        fn_copy_scaned_indices_from.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0); 
    else 
        fn_launch_function_in_loop(curr_loop - 1, fn_copy_scaned_indices_from);
        // fn_copy_scaned_indices_from.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, curr_loop - 1);
}
void RandomGraphColoringGPU::init_palette()
{
    get_command_list().add_task(fn_init_palette);
    fn_init_palette.bind_ptr(vivace_data->uncolored_verts_indirect_cmd_buffer); 
    fn_init_palette.bind_ptr(vivace_data->uncolored_verts_count); 
    fn_init_palette.bind_ptr(vivace_data->uncolored_verts); 
    fn_init_palette.bind_ptr(self_collision_data->collision_count); 
    fn_init_palette.bind_ptr(vivace_data->num_colors_self_collision); 
    fn_init_palette.bind_ptr(vivace_data->P_v); 
    fn_init_palette.bind_ptr(vivace_data->P_v_prev); 
    fn_init_palette.bind_ptr(vivace_data->next_color);
    fn_init_palette.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, 0);
    // fn_launch_function_in_loop(0, fn_init_palette);
}
void RandomGraphColoringGPU::tentative_coloring(const uint curr_loop)
{
    get_command_list().add_task(fn_tentative_coloring);
    fn_tentative_coloring.bind_ptr(vivace_data->uncolored_verts_count); 
    fn_tentative_coloring.bind_ptr(vivace_data->uncolored_verts); 
    fn_tentative_coloring.bind_ptr(vivace_data->num_colors_self_collision); 
    fn_tentative_coloring.bind_ptr(vivace_data->P_v); 
    fn_tentative_coloring.bind_ptr(vivace_data->P_v_prev); 
    fn_tentative_coloring.bind_ptr(vivace_data->next_color); 
    fn_tentative_coloring.bind_ptr(vivace_data->c_v); 
    fn_tentative_coloring.bind_ptr(vivace_data->pre_computed_random_number_256); 
    fn_tentative_coloring.bind_ptr(self_collision_data->collision_count); 
    fn_tentative_coloring.bind_constant(curr_loop); 
    fn_launch_function_in_loop(curr_loop, fn_tentative_coloring);
    // fn_tentative_coloring.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, curr_loop);
}
void RandomGraphColoringGPU::conflict_resolution_per_element_vv(const uint curr_loop)
{
    get_command_list().add_task(fn_copy_colred);   
    fn_copy_colred.bind_ptr(vivace_data->colored_in_curr_pass);
    fn_copy_colred.bind_ptr(vivace_data->colored);
    fn_copy_colred.bind_ptr(self_collision_data->collision_count);
    fn_copy_colred.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0);


    get_command_list().add_task(fn_conflict_resolution_vv);    
    fn_conflict_resolution_vv.bind_ptr(self_collision_data->narrow_phase_list_indices_vv);
    fn_conflict_resolution_vv.bind_ptr(self_collision_data->vert_VV_num_narrow_phase);
    fn_conflict_resolution_vv.bind_ptr(self_collision_data->vert_VV_prefix_narrow_phase);
    fn_conflict_resolution_vv.bind_ptr(self_collision_data->vert_adj_elements);

    fn_conflict_resolution_vv.bind_ptr(vivace_data->uncolored_verts_count);
    fn_conflict_resolution_vv.bind_ptr(vivace_data->uncolored_verts);

    fn_conflict_resolution_vv.bind_ptr(vivace_data->P_v);
    fn_conflict_resolution_vv.bind_ptr(vivace_data->c_v);
    fn_conflict_resolution_vv.bind_ptr(vivace_data->colored);
    fn_conflict_resolution_vv.bind_ptr(vivace_data->colored_in_curr_pass);

    fn_conflict_resolution_vv.bind_ptr(vivace_data->verts_prefix_in_cluster);
    fn_conflict_resolution_vv.bind_ptr(vivace_data->num_verts_in_cluster);

    fn_conflict_resolution_vv.bind_constant(curr_loop); 
    fn_launch_function_in_loop(curr_loop, fn_conflict_resolution_vv);



    get_command_list().add_task(fn_update_palatte_from_current_tentative_coloring_result_vv);    
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(self_collision_data->narrow_phase_list_indices_vv);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(self_collision_data->vert_VV_num_narrow_phase);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(self_collision_data->vert_VV_prefix_narrow_phase);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(self_collision_data->vert_adj_elements);

    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->uncolored_verts_count);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->uncolored_verts);

    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->P_v);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->c_v);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->colored);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->colored_in_curr_pass);

    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->verts_prefix_in_cluster);
    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_ptr(vivace_data->num_verts_in_cluster);

    fn_update_palatte_from_current_tentative_coloring_result_vv.bind_constant(curr_loop); 
    fn_launch_function_in_loop(curr_loop, fn_update_palatte_from_current_tentative_coloring_result_vv);
    
    
}
void RandomGraphColoringGPU::conflict_resolution_per_element_vf(const uint curr_loop)
{
    get_command_list().add_task(fn_copy_colred);   
    fn_copy_colred.bind_ptr(vivace_data->colored_in_curr_pass);
    fn_copy_colred.bind_ptr(vivace_data->colored);
    fn_copy_colred.bind_ptr(self_collision_data->collision_count);
    fn_copy_colred.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0);

    get_command_list().add_task(fn_conflict_resolution_vf);    
    fn_conflict_resolution_vf.bind_ptr(self_collision_data->narrow_phase_list_indices_vf);
    fn_conflict_resolution_vf.bind_ptr(self_collision_data->vert_VV_num_narrow_phase);
    fn_conflict_resolution_vf.bind_ptr(self_collision_data->vert_VV_prefix_narrow_phase);
    fn_conflict_resolution_vf.bind_ptr(self_collision_data->vert_adj_elements);

    fn_conflict_resolution_vf.bind_ptr(vivace_data->uncolored_verts_count);
    fn_conflict_resolution_vf.bind_ptr(vivace_data->uncolored_verts);

    fn_conflict_resolution_vf.bind_ptr(vivace_data->P_v);
    fn_conflict_resolution_vf.bind_ptr(vivace_data->c_v);
    fn_conflict_resolution_vf.bind_ptr(vivace_data->colored);
    fn_conflict_resolution_vf.bind_ptr(vivace_data->colored_in_curr_pass);

    fn_conflict_resolution_vf.bind_ptr(vivace_data->verts_prefix_in_cluster);
    fn_conflict_resolution_vf.bind_ptr(vivace_data->num_verts_in_cluster);

    fn_conflict_resolution_vf.bind_constant(curr_loop); 
    fn_launch_function_in_loop(curr_loop, fn_conflict_resolution_vf);
}

void RandomGraphColoringGPU::feed_the_hungry(const uint curr_loop)
{
    get_command_list().add_task(fn_feed_the_hungry);    
    fn_feed_the_hungry.bind_ptr(vivace_data->uncolored_verts);
    fn_feed_the_hungry.bind_ptr(vivace_data->P_v);
    fn_feed_the_hungry.bind_ptr(self_collision_data->collision_count);
    fn_feed_the_hungry.bind_ptr(vivace_data->next_color);
    fn_feed_the_hungry.bind_ptr(vivace_data->uncolored_verts_indirect_cmd_buffer); 
    fn_feed_the_hungry.bind_ptr(vivace_data->uncolored_verts_count); 
    fn_feed_the_hungry.bind_ptr(vivace_data->num_colors_self_collision); 
    fn_feed_the_hungry.bind_constant(curr_loop); 
    fn_launch_function_in_loop(curr_loop, fn_feed_the_hungry);
    // fn_feed_the_hungry.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, 
    //                                                      curr_loop - 1, 256); // Still Needs Previous Indirect Command Information...
    
}
void RandomGraphColoringGPU::put_rest_vertices_into_additional_color()
{
    // const uint max_loop = 20;
    // get_command_list().add_task(fn_put_rest_vertices_into_additional_color);        
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->uncolored_verts_count);
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->uncolored_verts);
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->P_v); 
    // fn_put_rest_vertices_into_additional_color.bind_ptr(self_collision_data->collision_count);
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->pre_computed_random_number_256);
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->num_colors_self_collision_vv);
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->num_verts_in_cluster); 
    // fn_put_rest_vertices_into_additional_color.bind_ptr(vivace_data->clusterd_constraint_self_collision);
    // fn_put_rest_vertices_into_additional_color.bind_constant(max_loop); 
    // fn_put_rest_vertices_into_additional_color.bind_constant(cloth_data->num_verts_total); 
    // fn_put_rest_vertices_into_additional_color.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, max_loop);
}
void RandomGraphColoringGPU::put_rest_vertices_into_random_color()
{
    const uint max_loop = VivaceGraphCloring::max_graph_coloring_colors;
    get_command_list().add_task(fn_put_rest_vertices_into_random_color);        
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->uncolored_verts_count);
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->uncolored_verts);
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->P_v); 
    fn_put_rest_vertices_into_random_color.bind_ptr(self_collision_data->collision_count);
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->pre_computed_random_number_256);
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->num_colors_self_collision);
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->num_verts_in_cluster); 
    fn_put_rest_vertices_into_random_color.bind_ptr(vivace_data->clusterd_constraint_self_collision);
    fn_put_rest_vertices_into_random_color.bind_constant(max_loop); 
    fn_launch_function_in_loop(max_loop, fn_put_rest_vertices_into_random_color);
    // fn_put_rest_vertices_into_random_color.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, max_loop);
}
void RandomGraphColoringGPU::make_cluster_indirect_cmd_buffer()
{
    get_command_list().add_task(fn_scan_num_verts_in_color);
    fn_scan_num_verts_in_color.bind_ptr(vivace_data->num_verts_in_cluster);
    fn_scan_num_verts_in_color.bind_ptr(vivace_data->cluster_prefix);
    fn_scan_num_verts_in_color.bind_ptr(vivace_data->clusterd_constraint_self_collision_indirect_cmd_buffer);
    fn_scan_num_verts_in_color.launch_async(64, 64);

    get_command_list().add_task(fn_fill_in_cluster_indices_vv);
    fn_fill_in_cluster_indices_vv.bind_ptr(vivace_data->verts_prefix_in_cluster); 
    fn_fill_in_cluster_indices_vv.bind_ptr(vivace_data->c_v);
    fn_fill_in_cluster_indices_vv.bind_ptr(vivace_data->cluster_prefix);
    fn_fill_in_cluster_indices_vv.bind_ptr(vivace_data->clusterd_constraint_self_collision);
    fn_fill_in_cluster_indices_vv.bind_ptr(self_collision_data->collision_count);
    fn_fill_in_cluster_indices_vv.bind_ptr(self_collision_data->narrow_phase_list_pair_vv);
    fn_fill_in_cluster_indices_vv.bind_ptr(self_collision_data->narrow_phase_list_pair_vv_merged);
    fn_fill_in_cluster_indices_vv.launch_async(self_collision_data->self_collision_indirect_cmd_buffer, 0);

    if constexpr (false)
    {
        get_command_list().send_and_wait();
        auto* ptr = self_collision_data->collision_count.data();

        uint sum = 0; for (uint color = 0; color < VivaceGraphCloring::max_graph_coloring_colors; color++) sum += vivace_data->num_verts_in_cluster[color];
        fast_format_single("   In Susbstep {:2} : Collision Pair Count = {}, Min/Max Degree = {}/{}, NumColor = {}, Sum = {}", 
                get_scene_params().current_substep, self_collision_data->collision_count[0], 
                VivaceGraphCloring::fn_get_min_degree(ptr), VivaceGraphCloring::fn_get_max_degree(ptr), 
                vivace_data->num_colors_self_collision[0], sum);
        fast_format_single(" , Remain Verts : {} ", ptr[0]);
        for (uint loop = 0; loop < VivaceGraphCloring::max_graph_coloring_colors; loop++)
        {
            fast_print_single(VivaceGraphCloring::fn_get_current_num_uncolored(vivace_data->uncolored_verts_count.data(), loop)); 
            if (vivace_data->uncolored_verts_count[loop] == 0) { fast_format_single(" ({}) ", loop + 1); break;}
        } fast_print();
    }
    

}

