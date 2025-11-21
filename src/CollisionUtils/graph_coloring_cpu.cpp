#include "graph_coloring_cpu.h"
#include "struct_to_string.h"
#include <set>

void RandomGraphColoringCPU::init_graph_coloring_system(VivaceColoringData &input_data, XpbdSelfCollision &input_self_collision) {
    vivace_data = &input_data;
    self_collision_data = &input_self_collision;

    {
        parallel_for(0, input_data.pre_computed_random_number_256.size(), [&](const uint i) {
            thread_local std::mt19937 generator(std::random_device{}());
            std::uniform_int_distribution<int> distribution(0, 255);
            uchar random_value = distribution(generator);
            input_data.pre_computed_random_number_256[i] = random_value;
            // VivaceGraphCloring::set_random_value_256(vid, input_data.pre_computed_random_number_256.data());
        });
    }
    // input_data.pre_computed_random_number_256.print([](const uchar value){ return std::format("{} ", uint(value)); });
}

void RandomGraphColoringCPU::graph_coloring_vivace() {
    //
    // Initialization
    //
    reduce_degree_and_set_max_color_from_max_degree();

    scan_uncolored_set(0);

    init_palette();

    uint max_loop = VivaceGraphCloring::max_graph_coloring_colors;

    for (uint curr_loop = 0; curr_loop < max_loop; curr_loop++) {
        const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
        if (curr_uncolored == 0) { break; }

        tentative_coloring(curr_loop);

        if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVV) { conflict_resolution_per_element_vv(curr_loop); }
        if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVF) { conflict_resolution_per_element_vf(curr_loop); }

        scan_uncolored_set(curr_loop + 1);

        feed_the_hungry(curr_loop + 1);
    }

    put_rest_vertices_into_random_color();

    make_cluster_indirect_cmd_buffer();
}

void RandomGraphColoringCPU::reduce_degree_and_set_max_color_from_max_degree() {
    Int2 min_max_degree = makeInt2(1000, 0);

    if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVV) {
        const uint num_collision = self_collision_data->collision_count[0];
        min_max_degree = parallel_for_and_reduce<Int2>(
            0, num_collision,
            [&](const uint element_id) {
                const uint degree = VivaceGraphCloring::reduce_degree_and_set_zero_degree_nodes_template(
                    element_id,
                    self_collision_data->narrow_phase_list_indices_vv.data(), self_collision_data->vert_VV_num_narrow_phase.data(),
                    vivace_data->uncolored_verts.data(), vivace_data->num_verts_in_cluster.data(), vivace_data->verts_prefix_in_cluster.data(),
                    vivace_data->colored.data(), vivace_data->c_v.data());
                return makeInt2(degree);
                // return VivaceGraphCloring::reduce_degree
            },
            VivaceGraphCloring::reduce_degree_binary_function,
            makeInt2(1000, 0));
    } else if (vivace_data->element_type == VivaceGraphColoringElementTypePerPairVF) {
        const uint num_collision = self_collision_data->collision_count[0];
        min_max_degree = parallel_for_and_reduce<Int2>(
            0, num_collision,
            [&](const uint element_id) {
                const uint degree = VivaceGraphCloring::reduce_degree_and_set_zero_degree_nodes_template(
                    element_id,
                    self_collision_data->narrow_phase_list_indices_vf.data(), self_collision_data->vert_VV_num_narrow_phase.data(),
                    vivace_data->uncolored_verts.data(), vivace_data->num_verts_in_cluster.data(), vivace_data->verts_prefix_in_cluster.data(),
                    vivace_data->colored.data(), vivace_data->c_v.data());
                return makeInt2(degree);
                // return VivaceGraphCloring::reduce_degree
            },
            VivaceGraphCloring::reduce_degree_binary_function,
            makeInt2(1000, 0));
    }
    // fast_format("   Readed Degree = {}", SimString::Vec2_to_string(min_max_degree));
    VivaceGraphCloring::set_max_color_from_max_degree(
        vivace_data->num_verts_in_cluster.data(), vivace_data->uncolored_verts_indirect_cmd_buffer.data(),
        vivace_data->num_colors_self_collision.data(),
        self_collision_data->collision_count.data(), min_max_degree);
}
void RandomGraphColoringCPU::scan_uncolored_set(const uint curr_loop) {
    auto &uncolored_verts_copy = vivace_data->clusterd_constraint_self_collision;
    const uint prev_uncolored = curr_loop == 0 ? self_collision_data->collision_count[0] : fn_get_current_uncolored_count(curr_loop - 1);

    // U \gets U - I
    parallel_for_and_scan(
        0, prev_uncolored,
        [&](const uint i) -> uint {
            return VivaceGraphCloring::scan_uncolored_set_CPU_1(i,
                                                                vivace_data->uncolored_verts.data(),
                                                                vivace_data->colored.data());
        },
        [&](const uint i, const uint &scan_result, const uint &self_result) {
            VivaceGraphCloring::scan_uncolored_set_CPU_2(i, scan_result, self_result,
                                                         vivace_data->uncolored_verts.data(),
                                                         uncolored_verts_copy.data(),
                                                         self_collision_data->collision_count.data(),
                                                         vivace_data->uncolored_verts_count.data(),
                                                         curr_loop);
        },
        0);

    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::copy_scaned_indices_from(i,
                                                     self_collision_data->collision_count.data(),
                                                     uncolored_verts_copy.data(),
                                                     vivace_data->uncolored_verts.data(),
                                                     vivace_data->uncolored_verts_count.data(),
                                                     vivace_data->uncolored_verts_indirect_cmd_buffer.data(),
                                                     curr_loop);
    });

    // uint sum = 0; for (uint color = 0; color < VivaceGraphCloring::max_graph_coloring_colors; color++) sum += vivace_data->num_verts_in_cluster[color];
    // fast_format("    Scan Unordered Set In loop {:2} : (Total = {}, Get {}) Uncolored = {}, Sum of Color = {} ", curr_loop, self_collision_data->collision_count[0],
    //     curr_uncolored + sum, curr_uncolored, sum);

    // fast_format("    Scan Unordered Set In loop {:2} = {} , new Uncolored = {}", curr_loop, prev_uncolored, curr_uncolored);
    // fast_format("      In Loop {} , Scan Vert From {} To {}", curr_loop, prev_uncolored, curr_uncolored);
}
void RandomGraphColoringCPU::init_palette() {
    const uint curr_uncolored = fn_get_current_uncolored_count(0);
    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::init_palette(i,
                                         vivace_data->uncolored_verts_indirect_cmd_buffer.data(),
                                         vivace_data->uncolored_verts_count.data(),
                                         vivace_data->uncolored_verts.data(),
                                         self_collision_data->collision_count.data(),
                                         vivace_data->num_colors_self_collision.data(),
                                         vivace_data->P_v.data(), vivace_data->P_v_prev.data(),
                                         vivace_data->next_color.data());
    });
}

void RandomGraphColoringCPU::tentative_coloring(const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
    if (curr_uncolored == 0) { return; }

    // if (curr_loop > 5)
    // {
    //     for (uint i = 0; i < curr_uncolored; i++)
    //     {
    //         const uint vid = vivace_data->uncolored_verts[i];
    //         if (!vivace_data->colored[vid])
    //         {
    //             if (curr_loop == 6) print_node_neighbor(vid);
    //             const uint valid_color_count = vivace_data->next_color[vid];
    //             auto Pv = vivace_data->P_v[vid] & make_lane_mask_64(valid_color_count);
    //             {
    //                 auto prev_Pv = vivace_data->P_v_prev[vid];
    //                 if ( (Pv & ~prev_Pv) == 0) { prev_Pv = 0; }
    //                 Pv &= ~prev_Pv;
    //             }
    //             const uint num_P_v = popc_uint64(Pv);
    //             const uint num_verts_total = self_collision_data->collision_count[0];
    //             uint read_value = vivace_data->pre_computed_random_number_256[(7 * curr_loop + vid) % num_verts_total];
    //             uint random_idx = read_value % num_P_v;
    //             uint curr_color;
    //             {
    //                 auto mask = Pv;
    //                 for (uint j = 0; j < random_idx; j++) // Drop Bits In Right Than random_idx
    //                     ffs_and_pop64(mask);
    //                 curr_color = ffs_uint64(mask) - 1;
    //             }
    //             fast_format("  {} : read random value = {}, random_idx = {} , Color = {} , Pv = {} (Popc = {})",
    //                 vid, read_value, random_idx, uint(vivace_data->c_v[vid]), SimString::bit_to_radix_string(vivace_data->P_v[vid], 16), num_P_v);
    //         }
    //     }
    // }

    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::tentative_coloring(i,
                                               vivace_data->uncolored_verts_count.data(), vivace_data->uncolored_verts.data(),
                                               vivace_data->num_colors_self_collision.data(),
                                               vivace_data->P_v.data(), vivace_data->P_v_prev.data(), vivace_data->next_color.data(), vivace_data->c_v.data(),
                                               vivace_data->pre_computed_random_number_256.data(),
                                               self_collision_data->collision_count.data(),
                                               curr_loop);
    });
}
void RandomGraphColoringCPU::conflict_resolution_per_vert_vv(const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
    if (curr_uncolored == 0) { return; }

    // parallel_for(0, curr_uncolored, [&](const uint i)
    // {
    //     VivaceGraphCloring::conflict_resolution_vv(i,
    //         vivace_data->uncolored_verts_count.data(), vivace_data->uncolored_verts.data(),
    //         vivace_data->vert_VV_num_narrow_phase.data(),
    //         vivace_data->vert_VV_prefix_narrow_phase.data(),
    //         vivace_data->vert_adj_elements.data(),
    //         vivace_data->P_v.data(), vivace_data->c_v.data(), vivace_data->colored.data(),
    //         vivace_data->clusterd_constraint_self_collision.data(), vivace_data->num_verts_in_cluster.data(),
    //         curr_loop, cloth_data->num_verts_total);
    // });
}
void RandomGraphColoringCPU::conflict_resolution_per_vert_vf(const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
    // if (curr_uncolored == 0)   { return; }

    // parallel_for(0, curr_uncolored, [&](const uint i)
    // {
    //     VivaceGraphCloring::conflict_resolution_vf(i,
    //         vivace_data->uncolored_verts_count.data(), vivace_data->uncolored_verts.data(),
    //         vivace_data->vert_VV_num_narrow_phase.data(),
    //         vivace_data->vert_VV_prefix_narrow_phase.data(),
    //         vivace_data->vert_adj_elements.data(),
    //         vivace_data->narrow_phase_list_vf.data(),
    //         vivace_data->P_v.data(), vivace_data->c_v.data(), vivace_data->colored.data(),
    //         vivace_data->clusterd_constraint_self_collision.data(), vivace_data->num_verts_in_cluster.data(),
    //         curr_loop, cloth_data->num_verts_total);
    // });
}
void RandomGraphColoringCPU::conflict_resolution_per_element_vv(const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
    if (curr_uncolored == 0) { return; }

    // parallel_copy(vivace_data->colored_copy.data(), vivace_data->colored.data(), self_collision_data->collision_count[0]);
    parallel_set(vivace_data->colored_in_curr_pass.data(), self_collision_data->collision_count[0], uchar(0));

    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::conflict_resolution_PerConstraint_template(i,
                                                                       self_collision_data->narrow_phase_list_indices_vv.data(),
                                                                       self_collision_data->vert_VV_num_narrow_phase.data(),
                                                                       self_collision_data->vert_VV_prefix_narrow_phase.data(),
                                                                       self_collision_data->vert_adj_elements.data(),

                                                                       vivace_data->uncolored_verts_count.data(),
                                                                       vivace_data->uncolored_verts.data(),

                                                                       vivace_data->P_v.data(), vivace_data->c_v.data(),
                                                                       vivace_data->colored.data(), vivace_data->colored_in_curr_pass.data(),

                                                                       vivace_data->verts_prefix_in_cluster.data(), vivace_data->num_verts_in_cluster.data(),
                                                                       curr_loop);
    });
    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::update_palatte_from_current_tentative_coloring_result_template(i,
                                                                                           self_collision_data->narrow_phase_list_indices_vv.data(),
                                                                                           self_collision_data->vert_VV_num_narrow_phase.data(),
                                                                                           self_collision_data->vert_VV_prefix_narrow_phase.data(),
                                                                                           self_collision_data->vert_adj_elements.data(),

                                                                                           vivace_data->uncolored_verts_count.data(),
                                                                                           vivace_data->uncolored_verts.data(),

                                                                                           vivace_data->P_v.data(), vivace_data->c_v.data(),
                                                                                           vivace_data->colored.data(), vivace_data->colored_in_curr_pass.data(),

                                                                                           vivace_data->verts_prefix_in_cluster.data(), vivace_data->num_verts_in_cluster.data(),
                                                                                           curr_loop);
    });
}
void RandomGraphColoringCPU::conflict_resolution_per_element_vf(const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);

    parallel_copy(vivace_data->colored_in_curr_pass.data(), vivace_data->colored.data(), self_collision_data->collision_count[0]);

    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::conflict_resolution_PerConstraint_template(i,
                                                                       self_collision_data->narrow_phase_list_indices_vf.data(),
                                                                       self_collision_data->vert_VV_num_narrow_phase.data(),
                                                                       self_collision_data->vert_VV_prefix_narrow_phase.data(),
                                                                       self_collision_data->vert_adj_elements.data(),

                                                                       vivace_data->uncolored_verts_count.data(),
                                                                       vivace_data->uncolored_verts.data(),

                                                                       vivace_data->P_v.data(), vivace_data->c_v.data(),
                                                                       vivace_data->colored.data(), vivace_data->colored_in_curr_pass.data(),

                                                                       vivace_data->verts_prefix_in_cluster.data(), vivace_data->num_verts_in_cluster.data(),
                                                                       curr_loop);
    });
}
void RandomGraphColoringCPU::feed_the_hungry(const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_uncolored_count(curr_loop);
    if (curr_uncolored == 0) { return; }

    parallel_for(0, curr_uncolored, [&](const uint i) {
        VivaceGraphCloring::feed_the_hungry(i,
                                            vivace_data->uncolored_verts.data(),
                                            vivace_data->P_v.data(),
                                            self_collision_data->collision_count.data(),
                                            vivace_data->next_color.data(),
                                            vivace_data->uncolored_verts_indirect_cmd_buffer.data(),
                                            vivace_data->uncolored_verts_count.data(),
                                            vivace_data->num_colors_self_collision.data(),
                                            curr_loop);
    });
}

void RandomGraphColoringCPU::put_rest_vertices_into_additional_color() {
    // const uint max_loop = 20;
    // const uint final_uncolored = fn_get_current_uncolored_count(max_loop);
    // parallel_for(0, final_uncolored, [&](const uint i)
    // {
    //     VivaceGraphCloring::put_rest_vertices_into_additional_color(i,
    //         vivace_data->uncolored_verts_count.data(), vivace_data->uncolored_verts.data(),
    //         vivace_data->P_v.data(), self_collision_data->collision_count.data(),
    //         vivace_data->pre_computed_random_number_256.data(),
    //         vivace_data->num_colors_self_collision_vv.data(),
    //         vivace_data->num_verts_in_cluster.data(),
    //         vivace_data->clusterd_constraint_self_collision.data(),
    //         max_loop, cloth_data->num_verts_total);
    // });
    // xpbd_data->collision.collision_count[50 + get_scene_params().current_substep] = final_uncolored;
}
void RandomGraphColoringCPU::put_rest_vertices_into_random_color() {
    const uint final_uncolored = fn_get_current_uncolored_count(VivaceGraphCloring::max_graph_coloring_colors);
    // single_thread_for(0, final_uncolored, [&](const uint i)
    // {
    //     const uint element_id = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, vivace_data->uncolored_verts.data());
    //     print_node_neighbor(element_id);
    // });

    if (final_uncolored != 0) fast_format("In Substep {:2} : Remain {} Collision Pair In {}", get_scene_params().current_substep, final_uncolored, self_collision_data->collision_count[0]);

    // parallel_for(0, final_uncolored, [&](const uint i)
    // {
    //     VivaceGraphCloring::put_rest_vertices_into_random_color(i,
    //     // put_rest_vertices_into_random_color_template(i,
    //         vivace_data->uncolored_verts_count.data(),
    //         vivace_data->uncolored_verts.data(),
    //         vivace_data->P_v.data(),
    //         self_collision_data->collision_count.data(),
    //         vivace_data->pre_computed_random_number_256.data(),
    //         vivace_data->num_colors_self_collision.data(),
    //         vivace_data->num_verts_in_cluster.data(),
    //         vivace_data->verts_prefix_in_cluster.data(),
    //         max_loop);
    // });
}
void RandomGraphColoringCPU::make_cluster_indirect_cmd_buffer() {
    const uint num_colors = vivace_data->num_colors_self_collision[0];
    const uint num_collisions = self_collision_data->collision_count[0];

    {
        ThreadGroup uint prefix = 0;
        for (uint i = 0; i < VivaceGraphCloring::max_graph_coloring_colors; i++) {
            const uint color = i;
            const uint num_verts = vivace_data->num_verts_in_cluster[color];
            vivace_data->clusterd_constraint_self_collision_indirect_cmd_buffer[i] = make_indirect_command_buffer(num_verts);
            vivace_data->cluster_prefix[color] = prefix;
            prefix += num_verts;
            // VivaceGraphCloring::make_cluster_indirect_cmd_buffer(i,
            //     vivace_data->num_verts_in_cluster.data(),
            //     vivace_data->clusterd_constraint_self_collision_indirect_cmd_buffer.data() );
        }
    }

    parallel_for(0, num_collisions, [&](const uint element_id) {
        VivaceGraphCloring::fill_in_cluster_indices(element_id,
                                                    vivace_data->verts_prefix_in_cluster.data(),
                                                    vivace_data->c_v.data(),
                                                    vivace_data->cluster_prefix.data(),
                                                    vivace_data->clusterd_constraint_self_collision.data(),
                                                    self_collision_data->narrow_phase_list_pair_vv.data(),
                                                    self_collision_data->narrow_phase_list_pair_vv_merged.data());
    });

    // {
    //     single_thread_for(0, self_collision_data->collision_count[0], [&](const uint i)
    //     {
    //         print_node_neighbor(i);
    //     });
    // }

    // const uint actually_max_degree = parallel_for_and_reduce_max<uint>(0, self_collision_data->collision_count[0], [&](const uint element_id)
    // {
    //     return VivaceGraphCloring::reduce_degree_and_set_zero_degree_nodes_template(
    //                 element_id,
    //                 self_collision_data->narrow_phase_list_indices_vv.data(), self_collision_data->vert_VV_num_narrow_phase.data(),
    //                 vivace_data->uncolored_verts.data(), vivace_data->num_verts_in_cluster.data(), vivace_data->verts_prefix_in_cluster.data(),
    //                 vivace_data->colored.data(), vivace_data->c_v.data());
    // });

    // auto* ptr = self_collision_data->collision_count.data();
    // fast_format_single("         In Susbstep {:2} : Collision Count = {}, Min/Max Degree = {}/{}, NumColor = {}",
    //         get_scene_params().current_substep, self_collision_data->collision_count[0],
    //         VivaceGraphCloring::fn_get_min_degree(ptr), actually_max_degree, vivace_data->num_colors_self_collision[0]);
    // fast_format_single(" , Remain Verts : {} ", ptr[0]);
    // for (uint loop = 0; loop < 20; loop++)
    // {
    //     fast_print_single(VivaceGraphCloring::fn_get_current_num_uncolored(vivace_data->uncolored_verts_count.data(), loop));
    // } fast_print();

    if constexpr (false) {
        auto *ptr = self_collision_data->collision_count.data();
        uint sum = 0;
        for (uint color = 0; color < VivaceGraphCloring::max_graph_coloring_colors; color++) sum += vivace_data->num_verts_in_cluster[color];
        fast_format_single("   In Susbstep {:2} : Collision Pair Count = {}, Min/Max Degree = {}/{}, NumColor = {}, Sum = {}",
                           get_scene_params().current_substep, self_collision_data->collision_count[0],
                           VivaceGraphCloring::fn_get_min_degree(ptr), VivaceGraphCloring::fn_get_max_degree(ptr),
                           vivace_data->num_colors_self_collision[0], sum);
        fast_format_single(" , Remain Verts : {} ", ptr[0]);
        for (uint loop = 0; loop < VivaceGraphCloring::max_graph_coloring_colors; loop++) {
            fast_print_single(VivaceGraphCloring::fn_get_current_num_uncolored(vivace_data->uncolored_verts_count.data(), loop));
            if (vivace_data->uncolored_verts_count[loop] == 0) {
                fast_format_single(" ({}) ", loop + 1);
                break;
            }
        }
        fast_print();

        // fast_format_single("   In Susbstep {:2} : Collision Count = {}, Min/Max Degree = {}/{}, NumColor = {}",
        //         get_scene_params().current_substep, self_collision_data->collision_count[0],
        //         VivaceGraphCloring::fn_get_min_degree(ptr), VivaceGraphCloring::fn_get_max_degree(ptr), vivace_data->num_colors_self_collision[0]);
        // fast_format_single(" , Merge Into {} Color : ", num_colors);
        // for (uint color = 0; color < num_colors + 1; color++)
        // {
        //     fast_print_single(vivace_data->num_verts_in_cluster[color]);
        // } fast_print();
    }
    if constexpr (false)// Check Redundant & Conflict
    {

        for (int cluster_idx = 0; cluster_idx < num_colors; cluster_idx++) {
            std::unordered_map<uint, Int2> local_map;
            const uint curr_prefix = vivace_data->cluster_prefix[cluster_idx];
            const uint cluster_size = vivace_data->num_verts_in_cluster[cluster_idx];
            const uint *cluster = vivace_data->clusterd_constraint_self_collision.data() + curr_prefix;

            single_thread_for(0, cluster_size, [&](const uint i) {
                const uint pair_idx = cluster[i];

                const uint curr_color = vivace_data->c_v[pair_idx];
                if (curr_color != cluster_idx) {
                    const uint offset = vivace_data->verts_prefix_in_cluster[pair_idx];
                    const uint prefix = vivace_data->cluster_prefix[curr_color];
                    const uint index = prefix + offset;

                    fast_format_err("Wrong Fill In Colors Into Cluster : Pair = {} : Desire for {} , Get {} in {} (From {} to {})", pair_idx, curr_color, i, cluster_idx, curr_prefix, curr_prefix + cluster_size);
                    fast_format_err("Wrong Fill In Colors Into Cluster : Pair = {} : Prefix = {} , Offset = {}, index = {}", pair_idx, prefix, offset, index);

                    fast_format("Verts In Cluster {} : ", cluster_idx);
                    fast_format_single("    ");
                    for (uint j = 0; j < cluster_size; j++) { fast_format_single("{}", cluster[j]); }
                    fast_print();

                    for (uint color = 0; color < num_colors; color++) {
                        fast_format("    Prefix of Color {} = {}", color, vivace_data->cluster_prefix[color]);
                    }
                    exit(0);
                }

                const Int2 pair = self_collision_data->narrow_phase_list_indices_vv[pair_idx];
                if (local_map.contains(pair[0] || local_map.contains(pair[1]))) {
                    for (uint j = 0; j < 2; j++) {
                        if (local_map.contains(pair[j])) {
                            auto value = local_map[pair[j]];
                            uint another = value[1];
                            Int2 another_pair = self_collision_data->narrow_phase_list_indices_vv[another];
                            fast_format_err("   Exsit Conflict Pair In Color {} : {} => {} (Vert = {}/{}), Conflict To {} => {} ({}/{})",
                                            cluster_idx,
                                            i, pair_idx, pair[0], pair[1],
                                            value[0], another, another_pair[0], another_pair[1]);
                            print_node_neighbor(pair_idx);
                            print_node_neighbor(another);
                            exit(0);
                        }
                    }
                    // exit(0);
                }
                local_map.insert(std::make_pair(pair[0], makeInt2(i, pair_idx)));
                local_map.insert(std::make_pair(pair[1], makeInt2(i, pair_idx)));
            });
        }
    }
}

void RandomGraphColoringCPU::print_node_neighbor(const uint pair_idx) {
    auto pair = self_collision_data->narrow_phase_list_indices_vv[pair_idx];
    auto pv = vivace_data->P_v[pair_idx];
    auto pv_prev = vivace_data->P_v_prev[pair_idx];
    uint next_color = vivace_data->next_color[pair_idx];

    fast_format_err("    Adj Info of {} : cv = {} , nextColor = {} , Pv = {}, MaskPv = {} ({})   ({}/{})",
                    pair_idx, vivace_data->c_v[pair_idx], next_color,
                    SimString::bit_to_radix_string(pv), SimString::bit_to_radix_string(pv_prev),
                    popc_uint64(pv & (~pv_prev)), pair[0], pair[1]);
    for (uint j = 0; j < 2; j++) {
        const uint vert = pair[j];
        const uint num_adj = self_collision_data->vert_VV_num_narrow_phase[vert];
        const uint start_idx = self_collision_data->vert_VV_prefix_narrow_phase[vert];

        for (uint jj = 0; jj < num_adj; jj++) {
            const uint adj_pair_idx = self_collision_data->vert_adj_elements[start_idx + jj];
            auto adj_pair = self_collision_data->narrow_phase_list_indices_vv[adj_pair_idx];
            if (vivace_data->colored[adj_pair_idx]) {
                fast_format_err("        Vert {:4} : {} , cv = {} ",
                                vert, adj_pair_idx, vivace_data->c_v[adj_pair_idx]);
            } else {
                fast_format_err("        Vert {:4} : {} , cv = {} , Pv = {} ({})  ({}/{})",
                                vert, adj_pair_idx, vivace_data->c_v[adj_pair_idx], popc_uint64(vivace_data->P_v[adj_pair_idx]), SimString::bit_to_radix_string(vivace_data->P_v[adj_pair_idx]), adj_pair[0], adj_pair[1]);
            }
        }
    }
}