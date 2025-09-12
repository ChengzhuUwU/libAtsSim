#define METAL_CODE
#define SIM_USE_SIMD true

#include <metal_stdlib>
using namespace metal;
#include "../src/SharedDefine/float_n.h"
#include "../src/SharedDefine/float_n_n.h"
#include "../src/SharedDefine/gpu_algorism.h"
#include "../src/Solver/xpbd_constraints.h"


kernel void empty()
{
    
}

kernel void test_add_1(
    Pointer(uint) input_ptr,
    Constant(uint) desire_value,
    uint index [[thread_position_in_grid]]
)
{
    const uint input_value = input_ptr[0];
    if (input_value == desire_value)
    {
        input_ptr[0] += 1;
    }
}
kernel void test_sum(
    Pointer(Float3) input_ptr,
    Pointer(float) output_ptr,
    Constant(uint) write_pos,
    uint vid [[thread_position_in_grid]],
    threadgroup_ids
)
{
    const Float3 vec = input_ptr[vid];
    float value = sqrt_scalar(length_squared_vec(vec));

    const uint bid = vid / 256;
    reduce_add(value);
    if (tid == 0) (output_ptr[100 + bid] = value);
}
kernel void test_sum_2(
    Pointer(float) input_ptr,
    Constant(uint) write_pos,
    uint vid [[thread_position_in_grid]],
    threadgroup_ids
)
{
    float value = input_ptr[100 + vid];
    reduce_add(value);
    if (tid == 0) input_ptr[write_pos] = value;
}

kernel void reset_float(
    Pointer(float) lambda_aaa, 
    uint index [[thread_position_in_grid]]
)
{
    lambda_aaa[index] = 0.0f;
}
kernel void reset_bool(
    Pointer(uchar) ptr_mask,
    uint vid [[thread_position_in_grid]]
)
{
    ptr_mask[vid] = 0;
}
kernel void reset_uint(
    Pointer(uint) ptr_mask,
    uint vid [[thread_position_in_grid]]
)
{
    ptr_mask[vid] = 0;
}



kernel void predict_position(
	Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_vert_velocity, 
    Pointer(Float3) sa_iter_start_position,
    Pointer(Float3) sa_x_tilde,

    Constant(bool) predict_for_collision, 
    Pointer(Float3) sa_next_position,
    
	Pointer(float) sa_vert_mass, 
    Pointer(uchar) sa_is_fixed,
	Constant(float) substep_dt,
	Constant(bool) fix_scene,
    uint vid [[thread_position_in_grid]])
{
    Constrains::Core::predict_position(vid, 
        sa_iter_position, sa_vert_velocity, sa_iter_start_position, sa_x_tilde,
        predict_for_collision, sa_next_position,
        sa_vert_mass, sa_is_fixed, substep_dt, fix_scene);
}
kernel void update_velocity(
    Pointer(Float3) sa_vert_velocity, 
	Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_iter_start_position, 
    Pointer(Float3) sa_start_position, 
	Pointer(Float3) sa_start_velocity, 
	Constant(float) substep_dt,
	Constant(float) damping,
	Constant(bool) fix_scene,
    uint vid [[thread_position_in_grid]])
{
    Constrains::Core::update_velocity(vid, 
        sa_vert_velocity, sa_iter_position, sa_iter_start_position, 
        sa_start_position, sa_start_velocity,
        substep_dt, damping, fix_scene);
}




kernel void copy_from_A_to_B(
     const Pointer(Float3) bufferA,
     Pointer(Float3) bufferB, 
     uint vid [[thread_position_in_grid]]
)
{
    bufferB[vid] = bufferA[vid];
}
kernel void copy_from_A_to_B_and_C(
     const Pointer(Float3) bufferA,
     Pointer(Float3) bufferB, 
     Pointer(Float3) bufferC, 
     uint vid [[thread_position_in_grid]]
)
{
    bufferB[vid] = bufferA[vid];
    bufferC[vid] = bufferA[vid];
}
kernel void read_and_solve_conflict(
    GLOBAL Float3* sa_begin_position_self, // GPU
	GLOBAL Float3* sa_begin_position_other, // CPU
	GLOBAL Float3* sa_iter_position_self, // GPU
	GLOBAL Float3* sa_iter_position_other, // CPU
    Constant(float) stretch_bending_assemble_weight, 
    uint vid [[thread_position_in_grid]]
)
{
    Constrains::Core::read_and_solve_conflict(vid, 
        sa_begin_position_self,sa_begin_position_other,
        sa_iter_position_self, sa_iter_position_other, 
        stretch_bending_assemble_weight);
}


kernel void constraint_ground_collision(
    Pointer(SceneParams) scene_params, 
    Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_iter_start_position,
    Pointer(float) lambda_ground_collision,
	Pointer(float) sa_mass_inv,
    uint vid [[thread_position_in_grid]]
)
{
    Constrains::solve_ground_collision_template(vid, 
        scene_params, sa_iter_position, sa_iter_start_position, lambda_ground_collision, sa_mass_inv);
}
kernel void constraint_stretch_mass_spring(
    const Pointer(Float3) input_position, 
    Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_iter_start_position, 
    Pointer(ATOMIC_FLAG) sa_vert_mutex, 
    Pointer(float) lambda_stretch_mass_spring,
	Pointer(float) sa_vert_mass_inv, 
    Pointer(Int2) sa_edges, 
    Pointer(float) sa_edge_rest_state_length, 

    Pointer(uint) clusterd_constraint_stretch_mass_spring,
    Constant(bool) use_multi_color,
    Constant(uint) curr_color_index,
    Constant(float) stiffness_stretch_spring,
    Constant(float) substep_dt,
    Constant(bool) use_atomic,

    uint index [[thread_position_in_grid]])
{
    const uint eid = Constrains::Core::get_index_from_color(use_multi_color, index, curr_color_index, clusterd_constraint_stretch_mass_spring);

    {
        Constrains::solve_stretch_mass_spring_template(
            eid, input_position, sa_iter_position, sa_iter_start_position,
            sa_vert_mutex, lambda_stretch_mass_spring,
            sa_vert_mass_inv, sa_edges, sa_edge_rest_state_length, stiffness_stretch_spring, substep_dt, use_atomic);

    }
}
kernel void constraint_neohookean(
    const Pointer(Float3) input_position, 
    Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_start_position, 
    Pointer(ATOMIC_FLAG) sa_vert_mutex, 
	Pointer(float) lambda_tet_neohookean_hydrostatic_term, 
    Pointer(float) lambda_tet_neohookean_distortion_term, 
	Pointer(float) sa_vert_mass_inv, 
    Pointer(Int4) sa_tets, 
    Pointer(float) sa_tet_volumn, 
    Pointer(Float3x3) sa_Dm_inv,

    Pointer(uint) clusterd_constraint_neohookean,
    Constant(bool) use_multi_color,
    Constant(uint) curr_color_index,
    Constant(float) m_first_lame,
    Constant(float) m_second_lame,
    Constant(float) substep_dt,
    Constant(bool) use_atomic,
    uint index [[thread_position_in_grid]])
{
    const uint tet_id = Constrains::Core::get_index_from_color(use_multi_color, index, curr_color_index, clusterd_constraint_neohookean);

    {
        // Constrains::solve_tetrahedral_fem_StVK_template(
        //     tet_id, input_position, sa_iter_position, sa_start_position,
        //     sa_vert_mutex, lambda_tet_neohookean_hydrostatic_term, lambda_tet_neohookean_distortion_term,
        //     sa_vert_mass_inv, sa_tets, sa_tet_volumn, sa_Dm_inv,
        //     m_first_lame, m_second_lame, substep_dt, use_atomic);

        Constrains::solve_tetrahedral_fem_NeoHookean_template(
            tet_id, input_position, sa_iter_position, sa_start_position,
            sa_vert_mutex, lambda_tet_neohookean_hydrostatic_term, lambda_tet_neohookean_distortion_term,
            sa_vert_mass_inv, sa_tets, sa_tet_volumn, sa_Dm_inv, 
            m_first_lame, m_second_lame, substep_dt, use_atomic);
    }   
}
kernel void constraint_bending_quadratic(
    const Pointer(Float3) input_position, 
    Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_start_position, 
    Pointer(ATOMIC_FLAG) sa_vert_mutex, 
	Pointer(float) lambda_bending, 
	Pointer(float) sa_vert_mass_inv, 
    Pointer(Int4) sa_bending_edges, 
    Pointer(Float2) sa_bending_edge_adj_faces_area,
	Pointer(Float4x4) sa_bending_edge_Q, 
    Pointer(float) sa_bending_edge_rest_state_angle,

    Pointer(uint) clusterd_constraint_bending,
    Constant(bool) use_multi_color,
    Constant(uint) curr_color_index,
    Constant(float) stiffness_bending_quadratic,
    Constant(float) stiffness_bending_DBA,
    Constant(float) substep_dt,
    Constant(bool) use_atomic,
    uint index [[thread_position_in_grid]])
{
    const uint eid = Constrains::Core::get_index_from_color(use_multi_color, index, curr_color_index, clusterd_constraint_bending);

    {
        Constrains::solve_bending_quadratic_template(
            eid, input_position, sa_iter_position, sa_start_position,
            sa_vert_mutex, lambda_bending, 
            sa_vert_mass_inv, sa_bending_edges, sa_bending_edge_adj_faces_area, 
            sa_bending_edge_Q, sa_bending_edge_rest_state_angle, 
            stiffness_bending_quadratic, stiffness_bending_DBA, substep_dt, use_atomic);
    }   

}
kernel void constraint_bending_DAB(
    const Pointer(Float3) input_position, 
    Pointer(Float3) sa_iter_position, 
    Pointer(Float3) sa_start_position, 
    Pointer(ATOMIC_FLAG) sa_vert_mutex, 
	Pointer(float) lambda_bending, 
	Pointer(float) sa_vert_mass_inv, 
    Pointer(Int4) sa_bending_edges, 
    Pointer(Float2) sa_bending_edge_adj_faces_area, 
	Pointer(Float4x4) sa_bending_edge_Q, 
    Pointer(float) sa_bending_edge_rest_state_angle,

    Pointer(uint) clusterd_constraint_bending,
    Constant(bool) use_multi_color,
    Constant(uint) curr_color_index,
    Constant(float) stiffness_bending_quadratic,
    Constant(float) stiffness_bending_DBA,
    Constant(float) substep_dt,
    Constant(bool) use_atomic,
    uint index [[thread_position_in_grid]])
{
    const uint eid = Constrains::Core::get_index_from_color(use_multi_color, index, curr_color_index, clusterd_constraint_bending);

    {
        Constrains::solve_bending_DAB_template_v2(
            eid, input_position, sa_iter_position, sa_start_position,
            sa_vert_mutex, lambda_bending, 
            sa_vert_mass_inv, sa_bending_edges, sa_bending_edge_adj_faces_area, 
            sa_bending_edge_Q, sa_bending_edge_rest_state_angle, 
            stiffness_bending_quadratic, stiffness_bending_DBA, substep_dt, use_atomic);
    }  
}


kernel void chebyshev_step(
    Pointer(Float3) iter_position,
    Pointer(Float3) sa_prev_1_iter_position,
    Pointer(Float3) sa_prev_2_iter_position,
    Pointer(uchar) sa_is_active_collide_vert_cloth,
    Constant(float) omega,
    uint vid [[thread_position_in_grid]]
)
{
    {
        // Chebyshev Acceleration
        if (!sa_is_active_collide_vert_cloth[vid])
        {
            Float3 x_k = iter_position[vid];
            Float3 x_k_sub_1 = sa_prev_1_iter_position[vid];
            Float3 x_k_sub_2 = sa_prev_2_iter_position[vid];

            x_k = omega * x_k + (1.f - omega) * x_k_sub_2;
            // x_k = omega * (x_k - x_k_sub_2) + x_k_sub_2;
            iter_position[vid] = x_k;

            sa_prev_2_iter_position[vid] = x_k_sub_1;
            sa_prev_1_iter_position[vid] = x_k;
        }
    }
}






kernel void compute_energy_inertia(
    Pointer(float) energyPtr, Constant(uint) pcg_it, const Pointer(Float3) updatePosition, 
    
    const Pointer(SceneParams) scene_params, 
    Pointer(uchar) sa_is_fixed,
    Pointer(float) sa_vert_mass,
    Pointer(Float3) sa_x_tilde,
    
    uint vid [[thread_position_in_grid]],
    threadgroup_ids)
{
    float energy = Constrains::Energy::compute_energy_inertia(vid, 
        updatePosition, scene_params, sa_is_fixed, sa_vert_mass, 
        sa_x_tilde);
        
    reduce_add(energy);
    if(tid == 0) atomic_add(energyPtr[pcg_it], energy);
}

kernel void compute_energy_stretch_mass_spring(
    Pointer(float) energyPtr, Constant(uint) pcg_it, const Pointer(Float3) updatePosition,

    Pointer(Int2) sa_edges, 
    Pointer(float) sa_edge_rest_state_length, 
    Constant(float) stiffness_stretch_spring,

    uint eid [[thread_position_in_grid]],
    threadgroup_ids
)
{
    float energy = Constrains::Energy::compute_energy_stretch_mass_spring(
        eid, updatePosition, 
        sa_edges, sa_edge_rest_state_length, stiffness_stretch_spring);

    reduce_add(energy);
    if(tid == 0) atomic_add(energyPtr[pcg_it], energy);
}

kernel void compute_energy_bending(
    Pointer(float) energyPtr, Constant(uint) pcg_it, const Pointer(Float3) updatePosition,

    Pointer(Int4) sa_bending_edges, 
    // Pointer(Int2) sa_bending_edge_adj_faces, 
    // Pointer(float) sa_face_area, 
	Pointer(Float4x4) sa_bending_edge_Q, 
    Pointer(float) sa_bending_edge_rest_state_angle,

    Constant(float) stiffness_bending_quadratic,
    Constant(float) stiffness_bending_DBA,
    Constant(Constrains::BendingType) bending_type,
    Constant(bool) use_xpbd,

    uint eid [[thread_position_in_grid]],
    threadgroup_ids
)
{
    float energy = Constrains::Energy::compute_energy_bending(
        bending_type, eid, updatePosition, 
        sa_bending_edges, nullptr, nullptr, 
        sa_bending_edge_Q, sa_bending_edge_rest_state_angle, 
        stiffness_bending_quadratic, stiffness_bending_DBA, use_xpbd);

    reduce_add(energy);
    if(tid == 0) atomic_add(energyPtr[pcg_it], energy);
}
kernel void compute_energy_stress_neohookean(
    Pointer(float) energyPtr, Constant(uint) pcg_it, const Pointer(Float3) updatePosition,

    const Pointer(Int4) sa_tets, 
    const Pointer(Float3x3) sa_Dm_inv, 
    const Pointer(float) sa_tet_volumn,
    Constant(float) m_first_lame, 
    Constant(float) m_second_lame,

    uint tet_id [[thread_position_in_grid]],
    threadgroup_ids
    )
{
    float energy = Constrains::Energy::compute_energy_stress_neohookean(
        tet_id, updatePosition, 
        sa_tets, sa_Dm_inv, sa_tet_volumn,
        m_first_lame, m_second_lame);

    reduce_add(energy);
    if(tid == 0) atomic_add(energyPtr[pcg_it], energy);
}

kernel void compute_energy_collision_vv(
    Pointer(float) energyPtr, Constant(uint) pcg_it, const Pointer(Float3) updatePosition,

	Pointer(ProximityVV) collision_self_vv,
	Pointer(uint) collision_count,
    Constant(float) thickness,

    uint pair_idx [[thread_position_in_grid]],
    threadgroup_ids
)
{
    float energy = Constrains::Energy::compute_energy_collision_vv(
        pair_idx, updatePosition, 
        collision_self_vv, collision_count, 
        thickness);

    reduce_add(energy);
    if (tid == 0) atomic_add(energyPtr[pcg_it], energy);
}
kernel void compute_energy_collision_vf(
    Pointer(float) energyPtr, Constant(uint) pcg_it, const Pointer(Float3) updatePosition,

    const Pointer(Float3) obstaclePosition,
	Pointer(ProximityVF) collision_self_vf,
	Pointer(uint) collision_count,
    Constant(float) thickness,

    uint pair_idx [[thread_position_in_grid]],
    threadgroup_ids
)
{
    float energy = Constrains::Energy::compute_energy_collision_vf(
        pair_idx, updatePosition, obstaclePosition,
        collision_self_vf, collision_count, 
        thickness);

    reduce_add(energy);
    if (tid == 0) atomic_add(energyPtr[pcg_it], energy);
}








kernel void evaluate_inertia(
	Pointer(Float4x3) sa_hf, Pointer(Float3) sa_iter_position, 
	Pointer(Float3) sa_x_tilde,
	Pointer(uchar) sa_is_fixed, Pointer(float) sa_vert_mass, Pointer(SceneParams) scene_params,
	Constant(float) substep_dt,

    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

	sa_hf[vid] = Constrains::VBD::compute_inertia(
        vid, sa_iter_position, 
        sa_x_tilde,
        sa_is_fixed, sa_vert_mass, scene_params,
        substep_dt);
};
kernel void vbd_step(
    Pointer(Float4x3) sa_hf, 
    Pointer(Float3) sa_iter_position,

    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    const uint i [[thread_position_in_grid]]
    )
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

    Float4x3 Hf = sa_hf[vid];
    Float3x3 H; Float3 f;
    Constrains::VBD::extractHf(Hf, f, H);
    // Float3x3 H = makeFloat3x3(get(Hf, 0), get(Hf, 1), get(Hf, 2));
    // Float3 f = get(Hf, 3);
    float det = determinant_mat(H);
    if (abs_scalar(det) > Epsilon)
    {
        Float3x3 H_inv = inverse_mat(H, det);
        Float3 dx = H_inv * f;
        // if (cluster == -1u)
        // {
        //     const float alpha = 0.3f;
        //     dx *= alpha;
        // }
        sa_iter_position[vid] += dx;
    }
}
kernel void evaluate_stretch_mass_spring(
	Pointer(Float4x3) sa_hf, Pointer(Float3) sa_iter_position, 
	Pointer(uint) sa_vert_adj_edges_csr, 
	Pointer(Int2) sa_edges, Pointer(float) sa_rest_length,
	Constant(float) stiffness_stretch, 

    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

	sa_hf[vid] += Constrains::VBD::compute_stretch_mass_spring(
                vid, sa_iter_position,
                sa_vert_adj_edges_csr, sa_edges, 
                sa_rest_length,
                stiffness_stretch);

};
kernel void evaluate_bending(
	Pointer(Float4x3) sa_hf, 
    Pointer(Float3) sa_iter_position,
	Pointer(uint) sa_vert_adj_bending_edges_csr,
	Pointer(Int4) sa_bending_edges, Pointer(Float4x4) sa_bending_edge_Q, 
    Constant(float) stiffness_quadratic_bending,

    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

	sa_hf[vid] += Constrains::VBD::compute_bending_quadratic(
                vid, sa_iter_position,
                sa_vert_adj_bending_edges_csr, sa_bending_edges, 
                sa_bending_edge_Q, 
                stiffness_quadratic_bending);
};
kernel void  evaluate_ground_collision(
	Pointer(Float4x3) sa_hf, 
	Pointer(Float3) sa_iter_position, 
    Constant(float) stiffness_collision,
	Constant(float) thickness_vv_obstacle, 
    Pointer(SceneParams) scene_param,

    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    Pointer(uchar) sa_is_active_collide_vert_cloth,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

    auto Hf = Constrains::VBD::compute_ground_collision(
                vid, sa_iter_position,
                stiffness_collision, thickness_vv_obstacle, scene_param);
	sa_hf[vid] += Hf;
    if (length_squared_vec(get(Hf, 3)) > Epsilon)
    {
        sa_is_active_collide_vert_cloth[vid] = true;
    }

};
kernel void  evaluate_obstacle_collision(
	Pointer(Float4x3) sa_hf, 
    Pointer(Float3) sa_iter_position, Pointer(Float3) sa_obstacle_substep_position,
	Pointer(uint) vert_VV_prefix_narrow_phase, Pointer(uint) vert_VV_num_narrow_phase, Pointer(uint) vert_adj_elements,
	Pointer(ProximityVF) narrow_phase_list_pair_vf, 
    Constant(float) thickness_vv_obstacle, 
    Constant(float) stiffness_collision,

    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    Pointer(uchar) sa_is_active_collide_vert_cloth,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

	auto Hf = Constrains::VBD::compute_obstacle_collision(
                    vid, sa_iter_position, sa_obstacle_substep_position, 
                    vert_VV_prefix_narrow_phase, vert_VV_num_narrow_phase, 
                    vert_adj_elements, 
                    narrow_phase_list_pair_vf, 
                    thickness_vv_obstacle, stiffness_collision);
    sa_hf[vid] += Hf;
    if (length_squared_vec(get(Hf, 3)) > Epsilon)
    {
        sa_is_active_collide_vert_cloth[vid] = true;
    }
};
kernel void  evaluate_self_collision(
	Pointer(Float4x3) sa_hf, 
    Pointer(Float3) sa_iter_position, 
	Pointer(uint) vert_VV_prefix_narrow_phase, Pointer(uint) vert_VV_num_narrow_phase, Pointer(uint) vert_adj_elements,
	Pointer(ProximityVV) narrow_phase_list_pair_vv, 
    Constant(float) thickness_vv_cloth, 
    Constant(float) stiffness_collision,
	
    Pointer(uint) clusterd_per_vertex_bending,
    Constant(uint) cluster,
    Pointer(uchar) sa_is_active_collide_vert_cloth,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

    auto Hf = Constrains::VBD::compute_self_collision(
                    vid, sa_iter_position, 
                    vert_VV_prefix_narrow_phase, vert_VV_num_narrow_phase, 
                    vert_adj_elements, 
                    narrow_phase_list_pair_vv, 
                    thickness_vv_cloth, stiffness_collision);
    sa_hf[vid] += Hf;
    if (length_squared_vec(get(Hf, 3)) > Epsilon)
    {
        sa_is_active_collide_vert_cloth[vid] = true;
    }
};





kernel void debug(
    Pointer(Float3) iter_position,
    Pointer(float) sum_buffer,
    Constant(uint) idx,
    uint vid [[thread_position_in_grid]]
)
{
    Float3 pos = iter_position[vid];
    float value = length_squared_vec(pos);
    atomic_add(sum_buffer[idx], value);
}