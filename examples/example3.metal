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
    PTR(uint) input_ptr,
    CONSTANT(uint) desire_value,
    uint index [[thread_position_in_grid]]
)
{
    const uint input_value = input_ptr[0];
    if (input_value == desire_value)
    {
        input_ptr[0] += 1;
    }
}

kernel void reset_float(
    PTR(float) lambda_aaa, 
    uint index [[thread_position_in_grid]]
)
{
    lambda_aaa[index] = 0.0f;
}
kernel void reset_bool(
    PTR(uchar) ptr_mask,
    uint vid [[thread_position_in_grid]]
)
{
    ptr_mask[vid] = 0;
}
kernel void reset_uint(
    PTR(uint) ptr_mask,
    uint vid [[thread_position_in_grid]]
)
{
    ptr_mask[vid] = 0;
}



kernel void predict_position(
	PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_vert_velocity, 
    PTR(Float3) sa_iter_start_position,

    CONSTANT(bool) predict_for_collision, 
    PTR(Float3) sa_next_position,
    
	PTR(float) sa_vert_mass, 
    PTR(uchar) sa_is_fixed,
	CONSTANT(float) substep_dt,
	CONSTANT(bool) fix_scene,
    uint vid [[thread_position_in_grid]])
{
    Constrains::Core::predict_position(vid, 
        sa_iter_position, sa_vert_velocity, sa_iter_start_position, 
        predict_for_collision, sa_next_position,
        sa_vert_mass, sa_is_fixed, substep_dt, fix_scene);
}
kernel void update_velocity(
    PTR(Float3) sa_vert_velocity, 
	PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_iter_start_position, 
    PTR(Float3) sa_start_position, 
	PTR(Float3) sa_start_velocity, 
	CONSTANT(float) substep_dt,
	CONSTANT(float) damping,
	CONSTANT(bool) fix_scene,
    uint vid [[thread_position_in_grid]])
{
    Constrains::Core::update_velocity(vid, 
        sa_vert_velocity, sa_iter_position, sa_iter_start_position, 
        sa_start_position, sa_start_velocity,
        substep_dt, damping, fix_scene);
}




kernel void copy_from_A_to_B(
     const PTR(Float3) bufferA,
     PTR(Float3) bufferB, 
     uint vid [[thread_position_in_grid]]
)
{
    bufferB[vid] = bufferA[vid];
}
kernel void copy_from_A_to_B_and_C(
     const PTR(Float3) bufferA,
     PTR(Float3) bufferB, 
     PTR(Float3) bufferC, 
     uint vid [[thread_position_in_grid]]
)
{
    bufferB[vid] = bufferA[vid];
    bufferC[vid] = bufferA[vid];
}
kernel void read_and_solve_conflict(
    DEVICE Float3* sa_begin_position_self, // GPU
	DEVICE Float3* sa_begin_position_other, // CPU
	DEVICE Float3* sa_iter_position_self, // GPU
	DEVICE Float3* sa_iter_position_other, // CPU
    CONSTANT(float) stretch_bending_assemble_weight, 
    uint vid [[thread_position_in_grid]]
)
{
    Constrains::Core::read_and_solve_conflict(vid, 
        sa_begin_position_self,sa_begin_position_other,
        sa_iter_position_self, sa_iter_position_other, 
        stretch_bending_assemble_weight);
}


kernel void constraint_ground_collision(
    PTR(SceneParams) scene_params, 
    PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_iter_start_position,
    PTR(float) lambda_ground_collision,
	PTR(float) sa_mass_inv,
    uint vid [[thread_position_in_grid]]
)
{
    Constrains::solve_ground_collision_template(vid, 
        scene_params, sa_iter_position, sa_iter_start_position, lambda_ground_collision, sa_mass_inv);
}
kernel void constraint_stretch_mass_spring(
    const PTR(Float3) input_position, 
    PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_iter_start_position, 
    PTR(ATOMIC_FLAG) sa_vert_mutex, 
    PTR(float) lambda_stretch_mass_spring,
	PTR(float) sa_vert_mass_inv, 
    PTR(Int2) sa_edges, 
    PTR(float) sa_edge_rest_state_length, 

    PTR(uint) clusterd_constraint_stretch_mass_spring,
    CONSTANT(bool) use_multi_color,
    CONSTANT(uint) curr_color_index,
    CONSTANT(float) stiffness_stretch_spring,
    CONSTANT(float) substep_dt,
    CONSTANT(bool) use_atomic,

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
    const PTR(Float3) input_position, 
    PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_start_position, 
    PTR(ATOMIC_FLAG) sa_vert_mutex, 
	PTR(float) lambda_tet_neohookean_hydrostatic_term, 
    PTR(float) lambda_tet_neohookean_distortion_term, 
	PTR(float) sa_vert_mass_inv, 
    PTR(Int4) sa_tets, 
    PTR(float) sa_tet_volumn, 
    PTR(Float3x3) sa_Dm_inv,

    PTR(uint) clusterd_constraint_neohookean,
    CONSTANT(bool) use_multi_color,
    CONSTANT(uint) curr_color_index,
    CONSTANT(float) m_first_lame,
    CONSTANT(float) m_second_lame,
    CONSTANT(float) substep_dt,
    CONSTANT(bool) use_atomic,
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
    const PTR(Float3) input_position, 
    PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_start_position, 
    PTR(ATOMIC_FLAG) sa_vert_mutex, 
	PTR(float) lambda_bending, 
	PTR(float) sa_vert_mass_inv, 
    PTR(Int4) sa_bending_edges, 
    PTR(Float2) sa_bending_edge_adj_faces_area,
	PTR(Float4x4) sa_bending_edge_Q, 
    PTR(float) sa_bending_edge_rest_state_angle,

    PTR(uint) clusterd_constraint_bending,
    CONSTANT(bool) use_multi_color,
    CONSTANT(uint) curr_color_index,
    CONSTANT(float) stiffness_bending_quadratic,
    CONSTANT(float) stiffness_bending_DBA,
    CONSTANT(float) substep_dt,
    CONSTANT(bool) use_atomic,
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
    const PTR(Float3) input_position, 
    PTR(Float3) sa_iter_position, 
    PTR(Float3) sa_start_position, 
    PTR(ATOMIC_FLAG) sa_vert_mutex, 
	PTR(float) lambda_bending, 
	PTR(float) sa_vert_mass_inv, 
    PTR(Int4) sa_bending_edges, 
    PTR(Float2) sa_bending_edge_adj_faces_area, 
	PTR(Float4x4) sa_bending_edge_Q, 
    PTR(float) sa_bending_edge_rest_state_angle,

    PTR(uint) clusterd_constraint_bending,
    CONSTANT(bool) use_multi_color,
    CONSTANT(uint) curr_color_index,
    CONSTANT(float) stiffness_bending_quadratic,
    CONSTANT(float) stiffness_bending_DBA,
    CONSTANT(float) substep_dt,
    CONSTANT(bool) use_atomic,
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
    PTR(Float3) iter_position,
    PTR(Float3) sa_prev_1_iter_position,
    PTR(Float3) sa_prev_2_iter_position,
    PTR(uchar) sa_is_active_collide_vert_cloth,
    CONSTANT(float) omega,
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
    PTR(float) energyPtr, CONSTANT(uint) pcg_it, const PTR(Float3) updatePosition, 
    
    const PTR(SceneParams) scene_params, 
    PTR(uchar) sa_is_fixed,
    PTR(float) sa_vert_mass,
    PTR(Float3) sa_start_position,
    PTR(Float3) sa_vert_start_velocity,
    
    uint vid [[thread_position_in_grid]],
    threadgroup_ids)
{
    float energy = Constrains::Energy::compute_energy_inertia(vid, 
        updatePosition, scene_params, sa_is_fixed, sa_vert_mass, 
        sa_start_position, sa_vert_start_velocity);
        
    reduce_add(energy);
    if(tid == 0) atomic_add(energyPtr[pcg_it], energy);
}

kernel void compute_energy_stretch_mass_spring(
    PTR(float) energyPtr, CONSTANT(uint) pcg_it, const PTR(Float3) updatePosition,

    PTR(Int2) sa_edges, 
    PTR(float) sa_edge_rest_state_length, 
    CONSTANT(float) stiffness_stretch_spring,

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
    PTR(float) energyPtr, CONSTANT(uint) pcg_it, const PTR(Float3) updatePosition,

    PTR(Int4) sa_bending_edges, 
    // PTR(Int2) sa_bending_edge_adj_faces, 
    // PTR(float) sa_face_area, 
	PTR(Float4x4) sa_bending_edge_Q, 
    PTR(float) sa_bending_edge_rest_state_angle,

    CONSTANT(float) stiffness_bending_quadratic,
    CONSTANT(float) stiffness_bending_DBA,
    CONSTANT(Constrains::BendingType) bending_type,
    CONSTANT(bool) use_xpbd,

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
    PTR(float) energyPtr, CONSTANT(uint) pcg_it, const PTR(Float3) updatePosition,

    const PTR(Int4) sa_tets, 
    const PTR(Float3x3) sa_Dm_inv, 
    const PTR(float) sa_tet_volumn,
    CONSTANT(float) m_first_lame, 
    CONSTANT(float) m_second_lame,

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
    PTR(float) energyPtr, CONSTANT(uint) pcg_it, const PTR(Float3) updatePosition,

	PTR(ProximityVV) collision_self_vv,
	PTR(uint) collision_count,
    CONSTANT(float) thickness,

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
    PTR(float) energyPtr, CONSTANT(uint) pcg_it, const PTR(Float3) updatePosition,

    const PTR(Float3) obstaclePosition,
	PTR(ProximityVF) collision_self_vf,
	PTR(uint) collision_count,
    CONSTANT(float) thickness,

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
	PTR(Float4x3) sa_hf, PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_iter_start_position, PTR(Float3) sa_vert_velocity,
	PTR(uchar) sa_is_fixed, PTR(float) sa_vert_mass, PTR(SceneParams) scene_params,
	CONSTANT(float) substep_dt,

    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
    const uint i [[thread_position_in_grid]]
	)
{
    const uint vid = cluster == -1u ? i : clusterd_per_vertex_bending[clusterd_per_vertex_bending[cluster] + i];

	sa_hf[vid] = Constrains::VBD::compute_inertia(
        vid, sa_iter_position, 
        sa_iter_start_position, sa_vert_velocity, 
        sa_is_fixed, sa_vert_mass, scene_params,
        substep_dt);
};
kernel void vbd_step(
    PTR(Float4x3) sa_hf, 
    PTR(Float3) sa_iter_position,

    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
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
	PTR(Float4x3) sa_hf, PTR(Float3) sa_iter_position, 
	PTR(uint) sa_vert_adj_edges_csr, 
	PTR(Int2) sa_edges, PTR(float) sa_rest_length,
	CONSTANT(float) stiffness_stretch, 

    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
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
	PTR(Float4x3) sa_hf, 
    PTR(Float3) sa_iter_position,
	PTR(uint) sa_vert_adj_bending_edges_csr,
	PTR(Int4) sa_bending_edges, PTR(Float4x4) sa_bending_edge_Q, 
    CONSTANT(float) stiffness_quadratic_bending,

    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
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
	PTR(Float4x3) sa_hf, 
	PTR(Float3) sa_iter_position, 
    CONSTANT(float) stiffness_collision,
	CONSTANT(float) thickness_vv_obstacle, 
    PTR(SceneParams) scene_param,

    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
    PTR(uchar) sa_is_active_collide_vert_cloth,
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
	PTR(Float4x3) sa_hf, 
    PTR(Float3) sa_iter_position, PTR(Float3) sa_obstacle_substep_position,
	PTR(uint) vert_VV_prefix_narrow_phase, PTR(uint) vert_VV_num_narrow_phase, PTR(uint) vert_adj_elements,
	PTR(ProximityVF) narrow_phase_list_pair_vf, 
    CONSTANT(float) thickness_vv_obstacle, 
    CONSTANT(float) stiffness_collision,

    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
    PTR(uchar) sa_is_active_collide_vert_cloth,
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
	PTR(Float4x3) sa_hf, 
    PTR(Float3) sa_iter_position, 
	PTR(uint) vert_VV_prefix_narrow_phase, PTR(uint) vert_VV_num_narrow_phase, PTR(uint) vert_adj_elements,
	PTR(ProximityVV) narrow_phase_list_pair_vv, 
    CONSTANT(float) thickness_vv_cloth, 
    CONSTANT(float) stiffness_collision,
	
    PTR(uint) clusterd_per_vertex_bending,
    CONSTANT(uint) cluster,
    PTR(uchar) sa_is_active_collide_vert_cloth,
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
    PTR(Float3) iter_position,
    PTR(float) sum_buffer,
    CONSTANT(uint) idx,
    uint vid [[thread_position_in_grid]]
)
{
    Float3 pos = iter_position[vid];
    float value = length_squared_vec(pos);
    atomic_add(sum_buffer[idx], value);
}