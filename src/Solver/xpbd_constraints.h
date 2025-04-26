#pragma once

///
/// XPBD Constrains
///

#include "float_n_n.h"
#include "float_n.h"
#include "make_arguments.h"
#include "scene_params.h"
#include "atomic.h"
#include "bits_utils.h"
#include <large_matrix.h>
#include <fem_energy.h>
#include "collision_data.h"

#ifndef METAL_CODE
#include <set>
#include <unordered_set>
#include <format>
// #include <eigen3/Eigen/Dense>
#endif

namespace Constrains
{

// const float youngs_modulus = 1e4;
// const float poisson_ratio = 0.2f;
// constexpr float stiffness_spring = 1e4;
// constexpr float stiffness_quadratic_bending = 0.01;

namespace Core
{

inline float caulc_alpha_tilde(
	CONST(float) dt, CONST(float) stiff)
{
	return 1.f / (stiff * dt * dt);
}
inline float caulc_gamma_tilde(
	CONST(float) dt, CONST(float) damping, CONST(float) stiff)
{
	return damping / (stiff * dt);
}

template<uint NumVerts>
inline float get_delta_xpbd_template(
	THREAD Float3* dx,
	const THREAD Float3* gradient,
	const THREAD float* weight,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	// const float alpha_tilde = 1.f / (stiff * dt * dt);
	float denom = alpha_tilde;
	for (uint i = 0; i < NumVerts; i++) 
	{
		denom += weight[i] * length_squared_vec(gradient[i]);
	}

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier
	// lambda += delta_lambda;

	if (denom != 0.f)
	{
		for (uint i = 0; i < NumVerts; i++) 
		{
			dx[i] = delta_lambda * weight[i] * gradient[i];
		}
	}
	return delta_lambda;
}
template<uint NumVerts>
inline float get_delta_xpbd_template_with_damping(
	THREAD Float3* dx,
	const THREAD Float3* gradient,
	const THREAD Float3* h_vel,
	const THREAD float* weight,
	CONST(float) C, CONST(float) lambda, 
	CONST(float) alpha_tilde, CONST(float) gamma_tilde)
{
	float denom = 0.0f; float sum_h_vel_g = 0.0f;
	for (uint i = 0; i < NumVerts; i++) 
	{
		denom += weight[i] * length_squared_vec(gradient[i]);
		sum_h_vel_g += dot_vec(gradient[i], h_vel[i]);
	}
	denom *= (1.0f + gamma_tilde);
	denom += alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda - gamma_tilde * sum_h_vel_g) / (denom); // lagrange multiplier
	// lambda += delta_lambda;

	if (denom != 0.f)
	{
		for (uint i = 0; i < NumVerts; i++) 
		{
			dx[i] = delta_lambda * weight[i] * gradient[i];
		}
	}
	return delta_lambda;
}
inline float get_delta_xpbd_1(
	TREF(Float3) dx1, 
	CREF(Float3) gradient1, 
	CONST(float) weight1, 
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	const float denom = 
		weight1 * length_squared_vec(gradient1) + alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier

	if (denom != 0.f)
	{
		dx1 = delta_lambda * weight1 * gradient1;
	}
	return delta_lambda;
}
inline float get_delta_xpbd_2(
	TREF(Float3) dx1, TREF(Float3) dx2,
	CREF(Float3) gradient1, CREF(Float3) gradient2,
	CONST(float) weight1, CONST(float) weight2,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	// const float alpha_tilde = 1.f / (stiff * dt * dt);
	const float denom = 
		weight1 * length_squared_vec(gradient1) +
		weight2 * length_squared_vec(gradient2) + alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier
	// lambda += delta_lambda;

	if (denom != 0.f)
	{
		dx1 = delta_lambda * weight1 * gradient1;
		dx2 = delta_lambda * weight2 * gradient2;
	}
	return delta_lambda;
}
inline float get_additional_delta_xpbd_2(
	TREF(Float3) dx1, TREF(Float3) dx2,
	CREF(Float3) gradient1, CREF(Float3) gradient2,
	CONST(float) weight1, CONST(float) weight2,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	const float denom = 
		weight1 * length_squared_vec(gradient1) +
		weight2 * length_squared_vec(gradient2) + alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier

	if (denom != 0.f)
	{
		dx1 += delta_lambda * weight1 * gradient1;
		dx2 += delta_lambda * weight2 * gradient2;
	}
	return delta_lambda;
}
inline float get_additional_delta_xpbd_2_with_damping(
	TREF(Float3) dx1, TREF(Float3) dx2,
	CREF(Float3) gradient1, CREF(Float3) gradient2,
	CREF(Float3) h_vel_1, CREF(Float3) h_vel_2,
	CONST(float) weight1, CONST(float) weight2,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde, CONST(float) gamma_tilde)
{
	const float denom = 
		(weight1 * length_squared_vec(gradient1) +
		weight2 * length_squared_vec(gradient2)) * (1.0f + gamma_tilde) + alpha_tilde;
	const float sum_hvel_g = 
		dot_vec(gradient1, h_vel_1) +
		dot_vec(gradient2, h_vel_2);

	const float delta_lambda = (-C - alpha_tilde * lambda - gamma_tilde * sum_hvel_g) / (denom); // lagrange multiplier

	if (denom != 0.f)
	{
		dx1 += delta_lambda * weight1 * gradient1;
		dx2 += delta_lambda * weight2 * gradient2;
	}
	return delta_lambda;
}
inline float get_delta_xpbd_3(
	TREF(Float3) dx_0, TREF(Float3) dx_1, TREF(Float3) dx_2,
	CREF(Float3) gradient_0, CREF(Float3) gradient_1, CREF(Float3) gradient_2,
	CONST(float) w_0, CONST(float) w_1, CONST(float) w_2,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	// const float alpha_tilde = 1.f / (stiff * dt * dt);
	const float denom = 
		w_0 * length_squared_vec(gradient_0) +
		w_1 * length_squared_vec(gradient_1) +
		w_2 * length_squared_vec(gradient_2) + alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier
	// lambda += delta_lambda;

	if (denom != 0.f)
	{
		dx_0 = delta_lambda * w_0 * gradient_0;
		dx_1 = delta_lambda * w_1 * gradient_1;
		dx_2 = delta_lambda * w_2 * gradient_2;
	}
	return delta_lambda;
}
inline float get_delta_xpbd_4(
	TREF(Float3) dx_0, TREF(Float3) dx_1, TREF(Float3) dx_2, TREF(Float3) dx_3,
	CREF(Float3) gradient_0, CREF(Float3) gradient_1, CREF(Float3) gradient_2, CREF(Float3) gradient_3,
	CONST(float) w_0, CONST(float) w_1, CONST(float) w_2, CONST(float) w_3,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	// const float alpha_tilde = 1.f / (stiff * dt * dt);
	const float denom = 
		w_0 * length_squared_vec(gradient_0) +
		w_1 * length_squared_vec(gradient_1) +
		w_2 * length_squared_vec(gradient_2) +
		w_3 * length_squared_vec(gradient_3) + alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier
	// lambda += delta_lambda;

	if (denom != 0.f) 
	{
		dx_0 = delta_lambda * w_0 * gradient_0;
		dx_1 = delta_lambda * w_1 * gradient_1;
		dx_2 = delta_lambda * w_2 * gradient_2;
		dx_3 = delta_lambda * w_3 * gradient_3;
	}
	return delta_lambda;
}
inline float get_additional_delta_x_xpbd_4(
	TREF(Float3) dx_0, TREF(Float3) dx_1, TREF(Float3) dx_2, TREF(Float3) dx_3,
	CREF(Float3) gradient_0, CREF(Float3) gradient_1, CREF(Float3) gradient_2, CREF(Float3) gradient_3,
	CONST(float) w_0, CONST(float) w_1, CONST(float) w_2, CONST(float) w_3,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde)
{
	// const float alpha_tilde = 1.f / (stiff * dt * dt);
	const float denom = 
		w_0 * length_squared_vec(gradient_0) +
		w_1 * length_squared_vec(gradient_1) +
		w_2 * length_squared_vec(gradient_2) +
		w_3 * length_squared_vec(gradient_3) + alpha_tilde;

	const float delta_lambda = (-C - alpha_tilde * lambda) / (denom); // lagrange multiplier
	// lambda += delta_lambda;

	if (denom != 0.f) 
	{
		dx_0 += delta_lambda * w_0 * gradient_0;
		dx_1 += delta_lambda * w_1 * gradient_1;
		dx_2 += delta_lambda * w_2 * gradient_2;
		dx_3 += delta_lambda * w_3 * gradient_3;
	}
	return delta_lambda;
}
inline float get_additional_delta_x_xpbd_4_with_damping(
	TREF(Float3) dx_0, TREF(Float3) dx_1, TREF(Float3) dx_2, TREF(Float3) dx_3,
	CREF(Float3) gradient_0, CREF(Float3) gradient_1, CREF(Float3) gradient_2, CREF(Float3) gradient_3,
	CREF(Float3) hv_0, CREF(Float3) hv_1, CREF(Float3) hv_2, CREF(Float3) hv_3,
	CONST(float) w_0, CONST(float) w_1, CONST(float) w_2, CONST(float) w_3,
	CONST(float) C, CONST(float) lambda,
	CONST(float) alpha_tilde, CONST(float) gamma_tilde)
{
	// const float alpha_tilde = 1.f / (stiff * dt * dt);
	const float denom = 
		(w_0 * length_squared_vec(gradient_0) +
		w_1 * length_squared_vec(gradient_1) +
		w_2 * length_squared_vec(gradient_2) +
		w_3 * length_squared_vec(gradient_3)) * (1.0f + gamma_tilde) + alpha_tilde;
	const float sum_g_hv = 
		dot_vec(gradient_0, hv_0) +
		dot_vec(gradient_1, hv_1) +
		dot_vec(gradient_2, hv_2) +
		dot_vec(gradient_3, hv_3);

	const float delta_lambda = (-C - alpha_tilde * lambda - gamma_tilde * sum_g_hv) / (denom); // lagrange multiplier

	if (denom != 0.f) 
	{
		dx_0 += delta_lambda * w_0 * gradient_0;
		dx_1 += delta_lambda * w_1 * gradient_1;
		dx_2 += delta_lambda * w_2 * gradient_2;
		dx_3 += delta_lambda * w_3 * gradient_3;
	}
	return delta_lambda;
}

inline void get_force_from_xpbd_system_3(
	CREF(Float3) gradient_0, CREF(Float3) gradient_1, CREF(Float3) gradient_2,
	CONST(float) C, CONST(float) stiff,
	TREF(Float3) force_0, TREF(Float3) force_1, TREF(Float3) force_2)
{	
	force_0 = -stiff * C * gradient_0;
	force_1 = -stiff * C * gradient_1;
	force_2 = -stiff * C * gradient_2;
}




inline uint get_index_from_color(const bool use_multi_color, const uint kernel_index, const uint cluster_idx, PTR(uint) cluster)
{
	uint mapped_index = kernel_index;
	if (use_multi_color)
	{
		// uint curr_prefix = cluster[cluster_idx];
		// mapped_index = cluster[curr_prefix + kernel_index];
		uint curr_prefix = cluster[cluster_idx];
		mapped_index = curr_prefix + kernel_index;
	}
	return mapped_index;
}

inline void predict_position(
	const uint vid,
	PTR(SceneParams) scene_params,
	PTR(Float3) sa_iter_position, PTR(Float3) sa_vert_velocity, PTR(Float3) sa_iter_start_position, 
	const bool predict_for_collision, PTR(Float3) sa_next_position,
	PTR(float) sa_vert_mass, 
	PTR(uchar) sa_is_fixed,
	const float substep_dt,
	const bool fix_scene)
{
	// const Float3 gravity = scene_params->gravity;
	const Float3 gravity = makeFloat3(0, -9.8f, 0);
	// const float thick_ness = scene_params->thick_ness;
	// const float radius = 0.2f;
	
	// Float3 x_prev = sa_iter_position[vid];
	const Float3 x_prev = sa_iter_start_position[vid];
	const Float3 v_prev = sa_vert_velocity[vid];

	Float3 outer_acceleration = gravity; 

	
	// They are different due to floating point precision !!!!
	Float3 v_pred = v_prev + substep_dt * outer_acceleration; 
	// const Float3 v_pred = v_prev + makeFloat3(
	// 	substep_dt * outer_acceleration.x, 
	// 	substep_dt * outer_acceleration.y, 
	// 	substep_dt * outer_acceleration.z); 

	// If static problem, then their is no v_prev
	// Float3 v_pred = substep_dt * outer_acceleration; 

	// Velocity Filter
	{
		if (sa_is_fixed[vid]) { outer_acceleration = Zero3; v_pred = Zero3; }

		// const float max_vel = radius * thick_ness / sub_step_dt;
		// float len_vel = length_vec(vel_pred);
		// if(len_vel > max_vel) { vel_pred *= (vel_pred / max_vel); }
	}

	// if (predict_for_collision) 
	// {
	// 	float implicit_dt = scene_params->implicit_dt;
	// 	float self_collision_detection_frequece = scene_params->collision_detection_frequece;
	// 	float detection_dt = implicit_dt / self_collision_detection_frequece;
	// 	Float3 v_pred_for_collision = v_prev + detection_dt * outer_acceleration; 
	// 	Float3 x_pred_for_collision = x_prev + detection_dt * v_pred_for_collision;
	// 	sa_next_position[vid] = x_pred_for_collision;
	// }	
	
	const Float3 x_pred = x_prev + substep_dt * v_pred;
	
	// Equal To:
	//	 // float mass = sa_vert_mass[vid];
	//	 Float3 outer_force = mass * outer_acceleration;    
	//   Float3 x_k = x_0 + sub_step_dt * vel_0 + sub_step_dt * sub_step_dt / mass * outer_force;

	// for n particles do : 
	//    x_prev \gets x , v \gets v + h f_ext / m , x \gets x + hv
	// sa_iter_start_position[vid] = x_prev;
	// sa_vert_velocity[vid] = v_pred;
	sa_iter_position[vid] = x_pred;
}
inline void predict_position_vbd(
	const uint vid,
	PTR(SceneParams) scene_params,
	PTR(Float3) sa_iter_position, PTR(Float3) sa_iter_start_position, 
	PTR(Float3) sa_vert_velocity, PTR(Float3) sa_vert_prev_velocity,
	const bool predict_for_collision, PTR(Float3) sa_next_position,
	PTR(float) sa_vert_mass, 
	PTR(uchar) sa_is_fixed,
	const float h,
	const bool fix_scene)
{
	// const Float3 gravity = scene_params->gravity;
	// Float3 x_0 = sa_iter_start_position[vid];
	// Float3 v_0 = sa_vert_velocity[vid];
	// Float3 v_prev = sa_vert_prev_velocity[vid];

	// Float3 outer_acceleration = gravity; 

	// // Float3 v_pred = v_prev + substep_dt * outer_acceleration; 
	// Float3 x_k = x_0 + h * v_0;
	// {
	// 	// Adaptive Init
	// 	Float3 a_t = (v_0 - v_prev) / h;
	// 	float len_outer_accelaration = length_vec(outer_acceleration);

	// 	float a_t_ext = dot_vec(a_t, outer_acceleration / len_outer_accelaration);
	// 	float a_hat = a_t_ext > len_outer_accelaration ? 1.f 
	// 				: a_t_ext < 0 ?                      0.f
	// 				: a_t_ext / len_outer_accelaration;
	// 	x_k += a_hat * h * h * outer_acceleration;
	// }

	// // Velocity Filter
	// {
	// 	if (sa_is_fixed[vid]){ outer_acceleration = Zero3; x_k = x_0; }
	// }

	// sa_iter_position[vid] = x_k;
}

inline void update_obstacle_position_in_substep(const uint vid,
	PTR(Float3) sa_start_position, 
	PTR(Float3) sa_next_position,
	PTR(Float3) sa_substep_position, 
	PTR(Float3) sa_substep_velocity, 
	const float alpha, const float substep_dt)
{
	const Float3 start_pos = sa_start_position[vid];
	const Float3 next_pos = sa_next_position[vid];
	const Float3 delta = alpha * (next_pos - start_pos);
	sa_substep_position[vid] = start_pos + delta; // alpha *  + (1.0f - alpha) * ;
	// sa_substep_velocity[vid] = delta / substep_dt;
}
inline void update_obstacle_normal_in_substep(const uint fid,
	PTR(Int3) sa_faces, 
	PTR(Float3) sa_substep_position, 
	PTR(Float3) sa_face_normal)
{
	const Int3 face = sa_faces[fid];
	const Float3 vert_pos[3] = {
		sa_substep_position[face[0]],
		sa_substep_position[face[1]],
		sa_substep_position[face[2]]
	};
	sa_face_normal[fid] = compute_face_normal(
            vert_pos[0], 
            vert_pos[1], 
            vert_pos[2]);
}
// inline void attach_fixpoint_to_obstacle(
// 	const uint vid,
// 	PTR(Float3) sa_iter_position1,
// 	PTR(Float3) sa_iter_position2,
// 	PTR(Float3) sa_substep_position_obstacle,
// 	PTR(XpbdData::FixPointAttach) sa_fixpoint_attach_info)
// {
// 	auto attach = sa_fixpoint_attach_info[vid];
// 	if (attach.attach_vid != -1u)
// 	{
// 		const Float3 obs_position = sa_substep_position_obstacle[attach.attach_vid];
// 		const Float3 dx = attach.get_dx();
// 		sa_iter_position1[vid] = obs_position + dx;
// 		sa_iter_position2[vid] = obs_position + dx;
// 	}
// }

inline void store_previous_position(
	const uint vid,
	PTR(Float3) sa_iter_position, PTR(Float3) sa_prev_1_iter_position, PTR(Float3) sa_prev_2_iter_position)
{
	Float3 prev_pos = sa_prev_1_iter_position[vid];
	Float3 curr_pos = sa_iter_position[vid];
	sa_prev_2_iter_position[vid] = prev_pos;
	sa_prev_1_iter_position[vid] = curr_pos;
}

inline void under_relaxation_and_store_previous_position(
	const uint vid,
	const PTR(uint) sa_vert_adj_constraints,
	PTR(Float3) sa_iter_position, PTR(Float3) sa_prev_1_iter_position, PTR(Float3) sa_prev_2_iter_position)
{
	Float3 prev_pos = sa_prev_1_iter_position[vid];
	Float3 curr_pos = sa_iter_position[vid];

	const uint num_adj = sa_vert_adj_constraints[vid];
	if (num_adj != 1) 
	{
		Float3 dx = curr_pos - prev_pos;
		dx = (1.f / float(num_adj)) * dx;
		sa_iter_position[vid] = prev_pos + dx;
	}

	sa_prev_2_iter_position[vid] = prev_pos;
	sa_prev_1_iter_position[vid] = curr_pos;
}
inline void over_relaxation(
	const uint vid,
	const PTR(uint) sa_vert_adj_constraints,
	PTR(Float3) sa_iter_position, PTR(Float3) sa_prev_1_iter_position, PTR(Float3) sa_prev_2_iter_position)
{
	Float3 prev_pos = sa_prev_1_iter_position[vid];
	Float3 curr_pos = sa_iter_position[vid];

	const uint num_adj = sa_vert_adj_constraints[vid];
	if (num_adj != 1) 
	{
		Float3 dx = curr_pos - prev_pos;
		constexpr float under_ralaxation_factor = 1.9f;
		dx = (under_ralaxation_factor / float(num_adj)) * dx;
		sa_iter_position[vid] = prev_pos + dx;
	}
}
inline void under_relaxation(
	const uint vid,
	const PTR(uint) sa_vert_adj_constraints,
	PTR(Float3) sa_iter_position, PTR(Float3) sa_prev_1_iter_position)
{
	Float3 prev_pos = sa_prev_1_iter_position[vid];
	Float3 curr_pos = sa_iter_position[vid];

	const uint num_adj = sa_vert_adj_constraints[vid];
	if (num_adj != 1) 
	{
		Float3 dx = curr_pos - prev_pos;
		constexpr float under_ralaxation_factor = 1.9f;
		dx = (under_ralaxation_factor / float(num_adj)) * dx;
		sa_iter_position[vid] = prev_pos + dx;
	}
}

inline void update_velocity(
	const uint vid, 
	PTR(Float3) sa_vert_velocity, 
	PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_iter_start_position, 
	PTR(Float3) sa_start_position, 
	PTR(Float3) sa_velocity_copy, 
	const float substep_dt,
	const float damping,
	const bool fix_scene
)
{
	Float3 x_k_init = sa_iter_start_position[vid];
	Float3 x_k = sa_iter_position[vid];

	Float3 dx = x_k - x_k_init;
	Float3 vel = dx / substep_dt;

	if (fix_scene) 
	{
		dx = Zero3;
		vel = Zero3;
		sa_iter_position[vid] = sa_start_position[vid];
		return;
	}

	// const float damping = 6.f;
	vel *= exp(-damping * substep_dt);

	// Velocity Filter
	// {
	// 	float len_vel = length_vec(vel);
	// 	const float max_vel = 3.f;
	// 	if (len_vel > max_vel) 
	// 	{
	// 	    float alpha = max_vel / len_vel;
	// 	    vel = alpha * vel;
	// 	    x_k = x_k_init + alpha * dx;
	// 	    sa_iter_position[vid] = x_k;
	// 	}
	// }

	// Update Velocity
	sa_vert_velocity[vid] = vel;
	sa_velocity_copy[vid] = vel;
	sa_iter_start_position[vid] = x_k;
}
inline void velocity_damping_laplacian(
	const uint vid, 
	PTR(Float3) sa_start_velocity, 
	PTR(Float3) sa_vert_velocity, 
	PTR(uint) sa_vert_adj_verts,
	const float weight
)
{
	const Float3 orig_vel = sa_vert_velocity[vid];

	const uint curr_prefix = sa_vert_adj_verts[vid];
	const uint next_prefix = sa_vert_adj_verts[vid + 1];
	const uint num_adj = next_prefix - curr_prefix;

	Float3 sum_vel = Zero3;
	for (uint j = 0; j < num_adj; j++)
	{
		const uint adj_vid = sa_vert_adj_verts[curr_prefix + j];
		const Float3 adj_vel = sa_start_velocity[adj_vid];
		sum_vel += adj_vel;
	}
	sum_vel /= float(num_adj);

	sa_vert_velocity[vid] = weight * sum_vel + (1.0f - weight) * orig_vel;
}


inline void update_velocity_and_predict_position(const uint vid, 
	PTR(Float3) sa_vert_velocity, 
	PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_iter_start_position, 
	PTR(Float3) sa_start_position, 
	PTR(Float3) sa_start_velocity, 
	PTR(uchar) sa_is_fixed, 
	PTR(SceneParams) scene_params, 
	const float substep_dt,
	const bool fix_scene)
{
	Float3 x_k_init = sa_iter_start_position[vid];
	Float3 x_k = sa_iter_position[vid];

	Float3 dx = x_k - x_k_init;
	Float3 vel = dx / substep_dt;

	// if (fix_scene) 
	// {
	// 	dx = Zero3;
	// 	vel = Zero3;
	// 	sa_iter_position[vid] = sa_start_position[vid];
	// 	return;
	// }

	const float damping = 3.f;
	vel *= exp(-damping * substep_dt);

	const Float3 gravity = scene_params->gravity;
	Float3 outer_acceleration = gravity; 

	Float3 v_pred = vel + substep_dt * outer_acceleration; 

	// Velocity Filter
	{
		if(sa_is_fixed[vid]){ outer_acceleration = Zero3; v_pred = Zero3; }
	}
	
	Float3 x_pred = x_k + substep_dt * v_pred;
	sa_iter_position[vid] = x_pred;
}

inline void reset_constraints(
	const uint index,
	PTR(float) lambda_ground_collision_cloth,  
	PTR(float) lambda_ground_collision_tet,  
	PTR(float) lambda_stretch_mass_spring,  
	PTR(float) lambda_triangle_stretch_term,
	PTR(float) lambda_triangle_shear_term,  
	PTR(float) lambda_bending, 				
	PTR(float) lambda_tet_neohookean_hydrostatic_term, 				
	PTR(float) lambda_tet_neohookean_distortion_term,
	PTR(float) lambda_cloth_balloon,
	PTR(float) sa_cloth_volumn,
	const uint num_cloth, const uint num_verts_cloth, const uint num_verts_tet, const uint num_edges_total, const uint num_faces_total, const uint num_bending_edges_total, const uint num_tets_total
)
{
	if (index < num_verts_cloth)    		{ lambda_ground_collision_cloth[index] = 0.f; }
	if (index < num_verts_tet)	    		{ lambda_ground_collision_tet[index] = 0.f; }
	if (index < num_edges_total) 			{ lambda_stretch_mass_spring[index] = 0.f;  }
	if (index < num_faces_total) 			{ lambda_triangle_stretch_term[index] = 0.f; lambda_triangle_shear_term[index] = 0.f; }
	if (index < num_bending_edges_total) 	{ lambda_bending[index] = 0.f;  }
	if (index < num_tets_total) 			{ lambda_tet_neohookean_hydrostatic_term[index] = 0.f; lambda_tet_neohookean_distortion_term[index] = 0.f; }
	if (index < num_cloth) 					{ lambda_cloth_balloon[3 * index + 0] = 0; lambda_cloth_balloon[3 * index + 1] = 0; lambda_cloth_balloon[3 * index + 2] = 0; sa_cloth_volumn[2 * index + 1] = 0; }
}

template <typename ConstaintType, uint N = Meta::get_vec_length<ConstaintType>()>
inline void assemble_to_position(DEVICE Float3* sa_iter_position, DEVICE ATOMIC_FLAG* sa_vert_mutex, CREF(ConstaintType) constraint, THREAD Float3* dx, const bool use_atomic)
{
	if (use_atomic) 
	{
		#ifdef METAL_CODE
			PTR(ATOMIC_FLOAT) atomic_iter_position = (PTR(ATOMIC_FLOAT))sa_iter_position;
			for (uint i = 0; i < N; i++) 
			{
				atomic_add(atomic_iter_position, constraint[i], dx[i]);
			}
		#else
			for (uint i = 0; i < N; i++) 
			{
				atomic_add(sa_iter_position[constraint[i]], sa_vert_mutex[constraint[i]], dx[i]);
			}
		#endif
	}
	else 
	{
		for (uint i = 0; i < N; i++) 
		{
			sa_iter_position[constraint[i]] += (dx[i]);
		}
	}
}


inline void copy_position_to_2_devices(const uint vid, 
	DEVICE Float3* sa_iter_position,
	DEVICE Float3* sa_begin_position_cpu, // CPU
	DEVICE Float3* sa_begin_position_gpu, // GPU
	DEVICE Float3* sa_iter_position_cpu, // CPU
	DEVICE Float3* sa_iter_position_gpu // GPU)
)
{
	const Float3 pos = sa_iter_position[vid];
	sa_begin_position_cpu[vid] = pos;
	sa_begin_position_gpu[vid] = pos;
	sa_iter_position_cpu[vid] = pos;
	sa_iter_position_gpu [vid] = pos;
}

inline void assemble_result_from_2_devices(
	const uint vid, 
	DEVICE Float3* sa_iter_position,
	DEVICE Float3* sa_begin_position_cpu, // CPU
	DEVICE Float3* sa_begin_position_gpu, // GPU
	DEVICE Float3* sa_iter_position_cpu, // CPU
	DEVICE Float3* sa_iter_position_gpu, // GPU
	const float weight)
{
	Float3 pos_self = sa_iter_position_gpu[vid];
	Float3 pos_other = sa_iter_position_cpu[vid];

	const Float3 init_pos = sa_iter_position[vid];
	Float3 self_dx = pos_self - init_pos;
	Float3 anot_dx = pos_other - init_pos;
	
	{
		Float3 weighted_pos;

		if (length_squared_vec(self_dx) == 0)
		{
			weighted_pos = pos_other;
		} 
		else if (length_squared_vec(anot_dx) == 0)
		{
			weighted_pos = pos_self;
		}
		else
		{
			weighted_pos = init_pos + weight * (self_dx + anot_dx);
		}
		sa_iter_position[vid] = weighted_pos;
		sa_begin_position_cpu[vid] = weighted_pos;
		sa_begin_position_gpu[vid] = weighted_pos;
		// sa_iter_position_cpu[vid] = weighted_pos;
		// sa_iter_position_gpu[vid] = weighted_pos;

	}
	// Float3 new_pos = 0.5f * (pos1 + pos2);

	// CONSTIF (store_to_cpu)
	// 	sa_iter_position_cpu[vid] = new_pos;
	// else
	//  	sa_iter_position_gpu[vid] = new_pos;
	// sa_iter_position[vid] = new_pos;


	// // const Float3 init_pos = sa_iter_position[vid];
	// const Float3 dx_1 = sa_iter_position_cpu[vid] - sa_begin_position_cpu[vid];
	// const Float3 dx_2 = sa_iter_position_gpu[vid] - sa_begin_position_gpu[vid];

	// Float3 dx = Zero3;
	// if (length_squared_vec(dx_1) < Epsilon)
	// {
	// 	dx = dx_2;
	// }
	// else if (length_squared_vec(dx_1) < Epsilon) 
	// {
	// 	dx = dx_1;
	// }
	// else 
	// {
	// 	dx = weight * (dx_1 + dx_2);
	// }
	
	// Float3 weighted_pos;

	// CONSTIF (store_to_cpu)
	// {
	// 	weighted_pos = sa_iter_position_cpu[vid] + dx;
	// 	sa_iter_position_cpu[vid] = weighted_pos;
	// 	sa_begin_position_cpu[vid] = weighted_pos;
	// }
	// else                                                                                      
	// {
	// 	weighted_pos = sa_iter_position_gpu[vid] + dx;
	// 	sa_iter_position_gpu[vid] = weighted_pos;
	// 	sa_begin_position_gpu[vid] = weighted_pos;
	// }

	// sa_iter_position[vid] = weighted_pos;
	
}



inline void copy_current_position_to_2_constraints(
	const uint vid, const DEVICE Float3* sa_iter_position, 
	DEVICE Float3* sa_iter_position_1, 
	DEVICE Float3* sa_iter_position_2
	)
{
	const Float3 curr_pos = sa_iter_position[vid];
	sa_iter_position_1[vid] = curr_pos;
	sa_iter_position_2[vid] = curr_pos;
}

inline void read_and_solve_conflict(
	const uint vid, 
	DEVICE Float3* sa_begin_position_self, 
	DEVICE Float3* sa_begin_position_other,
	DEVICE Float3* sa_iter_position_self, 
	DEVICE Float3* sa_iter_position_other, 
	const float weight
	)
{
	Float3 pos_self = sa_iter_position_self[vid];
	Float3 pos_other = sa_iter_position_other[vid];

	if (weight > Epsilon)
	{
		const Float3 init_pos = sa_begin_position_self[vid];
		Float3 self_dx = pos_self - init_pos;
		Float3 anot_dx = pos_other - init_pos;

		Float3 weighted_pos = init_pos;

		// Bad
		// float len1 = length_vec(self_dx);
		// float len2 = length_vec(anot_dx);
		// if (len1 + len2 > 1e-8)
		// {
		// 	float inv_sum_len = 1.0f * (len1 + len2);
		// 	weighted_pos = init_pos + inv_sum_len * (len1 * self_dx + len2 * anot_dx);
		// }

		const bool active_a = length_squared_vec(self_dx) != 0;
		const bool active_b = length_squared_vec(anot_dx) != 0;
		if (active_a && active_b)
		{
			// weighted_pos = init_pos + (weight * self_dx + (1.0f - weight) * anot_dx);
			weighted_pos = init_pos + weight * (self_dx + anot_dx);
		}
		else if (active_a) 
		{
			weighted_pos = pos_self;
		}
		else if (active_b) 
		{	
			weighted_pos = pos_other;
		}
		sa_iter_position_self[vid] = weighted_pos;
		sa_begin_position_self[vid] = weighted_pos; // Can NOT Ignore!!!
	}
	else 
	{
		Float3 weighted_pos = 0.5f * (pos_self + pos_other);
		sa_iter_position_self[vid] = weighted_pos;
		sa_begin_position_self[vid] = weighted_pos;
	}

}


}

}



// Material Constraints
namespace Constrains
{

enum StretchType{
	StretchTypeMassSpring,
	StretchTypeStVK,
	StretchTypeBaraffWitkin,
};
enum BendingType{
	BendingTypeNone,
	BendingTypeQuadratic,
	BendingTypeDAB,
	BendingTypeMassSpring,
};

// dCdx Is The Inverse Direction To 'normal'
inline void get_collision_C_and_dCdx(CONST(float) C, CREF(Float3) normal, CONST(float) stiff, TREF(float) energy, TREF(Float3) gradient)
{

#define COLLISION_GRADIANT_MODE_ORIG_XPBD 1
#define COLLISION_GRADIANT_MODE_ENERGY_FORCE 2
#define COLLISION_GRADIANT_MODE 1

#if COLLISION_GRADIANT_MODE == COLLISION_GRADIANT_MODE_ORIG_XPBD
	{
		energy = C;
		gradient = -normal;
	}
#else
	{
		energy = 0.5f * stiff * C * C;
		gradient = -stiff * C * normal; 
	}
#endif
}


inline void solve_ground_collision_template(
	const uint vid, PTR(SceneParams) scene_params, 
	PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_iter_start_position,
	PTR(float) lambda_ground_collision,
	PTR(float) sa_mass_inv
	)
{
	Float3 x_k = sa_iter_position[vid];
	Float3 x_init = sa_iter_start_position[vid];

	if (scene_params->use_floor)
	{
		const float thickness = scene_params->thick_ness;
		const float floor_y = scene_params->floor.y + thickness;

		const Float3 normal = makeFloat3(0, 1, 0);
		// const Float3 diff = x_k - makeFloat3(x_k.x, floor_y, x_k.z);
		// const float proj = dot_vec(diff, normal);
		const float proj = x_k.y - floor_y;

		if (proj < thickness)
		{
			Float3 dx = Zero3;
			const float stiff = scene_params->xpbd_stiffness_collision;
			const float dt = scene_params->implicit_dt / float(scene_params->num_substep);
			const float alpha_tilde = 1.0f / (stiff * dt * dt);

			const float lambda_n = lambda_ground_collision[vid];

			// Non-Elastic Collision
			{
				const float C_n = thickness - proj;
				const float vert_mass_inv = sa_mass_inv[vid];

				float energy; Float3 gradient;
				get_collision_C_and_dCdx(C_n, normal, stiff, energy, gradient);
				
				const float delta_lambda = Constrains::Core::get_delta_xpbd_1(
					dx, gradient, vert_mass_inv, 
					energy, lambda_n, alpha_tilde);
				
				lambda_ground_collision[vid] += delta_lambda;
			}
			// Friction
			{
				const float friction_damping = 0.5f;
				Float3 dx_h = (x_k - x_init);
				dx -= friction_damping * dx_h;
			}

			sa_iter_position[vid] = x_k + dx;

		}
		// if(x_k.y < floor_y)
		// {
		// 	// Friction
		// 	{
		// 		const float damping = 0.8f;
		// 		Float3 diff = x_k - x_init;
		// 		x_k -= damping * diff;
		// 	}
		// 	x_k.y = floor_y ; // + thickness * 0.5f;
		// 	sa_iter_position[vid] = x_k;
		// }            
	}
}

inline void solve_stretch_mass_spring_template(
	const uint eid, const PTR(Float3) input_position, PTR(Float3) output_position, PTR(Float3) start_position,
	PTR(ATOMIC_FLAG) sa_vert_mutex, PTR(float) lambda_stretch_mass_spring,
	PTR(float) sa_vert_mass_inv, PTR(Int2) sa_edges, PTR(float) sa_edge_rest_state_length, 
	const float stiffness_spring, const float substep_dt, const bool use_atomic)
{
	Int2 edge = sa_edges[eid];

	float lambda = lambda_stretch_mass_spring[eid];
	Float3 dx[2] = {Zero3, Zero3};

	// Constrains::solve_DistanceConstraint(eid, edge, 
	//     input_position, sa_prev_1_iter_position.data(), sa_edge_rest_state_length.data(), sa_vert_mass_inv.data(),
	//     sub_step_dt, stiffness_stretch,
	//     lambda, delta_lambda, dx[0], dx[1]);

	float energy = 0.f;  // Float3 force[2] = {Zero3, Zero3};
	Float3 vert_pos[2] = {
		input_position[edge[0]],  
		input_position[edge[1]]  
	};
	float vert_inv_mass[2] = {
		sa_vert_mass_inv[edge[0]],  
		sa_vert_mass_inv[edge[1]]  
	};

	// Float3 velocity_mult_dt[2] = {
	// 	vert_pos[0] - start_position[edge[0]],  
	// 	vert_pos[1] - start_position[edge[1]]  
	// };

	const float rest_edge_length = sa_edge_rest_state_length[eid];

	{
		Float3 diff = vert_pos[1] - vert_pos[0];
		float orig_lengthsqr = length_squared_vec(diff);
		// float orig_lengthsqr = (diff.x * diff.x) + (diff.y * diff.y) + (diff.z * diff.z);
		if (orig_lengthsqr < 1e-8) { return; }
		float l = sqrt_scalar(orig_lengthsqr);
		// float l = max_scalar(sqrt_scalar(orig_lengthsqr), Epsilon);
		float l0 = rest_edge_length;
		float C = l - l0;
		Float3 dir = diff / l;
		Float3 g1 = -dir;
		// force[0] = -dir;
		// force[1] = -force[0];
		energy = C;

		const float alpha_tilde = stiffness_spring < Epsilon ? 1e9 : 1.f / (stiffness_spring * substep_dt * substep_dt);
		float denom = alpha_tilde + (vert_inv_mass[0] + vert_inv_mass[1]);
		const float delta_lambda = (-energy - alpha_tilde * lambda) / (denom); // lagrange multiplier
		if (denom != 0.f)
		{
			dx[0] =  delta_lambda * vert_inv_mass[0] * g1;
			dx[1] = -delta_lambda * vert_inv_mass[1] * g1;
		}
		lambda_stretch_mass_spring[eid] = lambda + delta_lambda; // zero-confict
	}
	{
		// MaterialForce::Core::stretch_mass_spring<true, true, false>(
		// 	&energy, force, nullptr, 
		// 	vert_pos, rest_edge_length, stiffness_spring);
		// force[0] = -force[0]; force[1] = -force[1]; // force = -gradient = -dE/dx
	}

	// const float alpha_tilde = stiffness_spring < Epsilon ? 1e9 : 1.f / (stiffness_spring * substep_dt * substep_dt);
	// const float delta_lambda = Core::get_delta_xpbd_template<2>(dx, force, vert_inv_mass, energy, lambda, alpha_tilde);
	
	// const float k_damping = 0.1; const float gamma_tilde = k_damping / (stiffness_spring * substep_dt);
	// const float delta_lambda = Core::get_delta_xpbd_template_with_damping<2>(dx, force, velocity_mult_dt, vert_inv_mass, energy, lambda, alpha_tilde, gamma_tilde);

	

	Core::assemble_to_position(output_position, sa_vert_mutex, edge, dx, use_atomic);

}


template <bool compute_energy, bool compute_gradient, bool compute_hessian>
inline void core_bending_quadratic(
    THREAD float* energy, THREAD Float3* force, THREAD Float3x3* hessian,
    const THREAD Float3 vert_pos[3],
    CREF(Float4x4) m_Q,   
    const float stiffness
)
{
    //
    // A quadratic bending model for inextensible surfaces
    //
    CONSTIF (compute_energy)
    {
        float sum = 0.f;
        for (uint ii = 0; ii < 4; ii++) 
        {
            for (uint jj = 0; jj < 4; jj++) 
            {
                // E_b = 1/2 (x^T)Qx = 1/2 Sigma_ij Q_ij <x_i, x_j>
                sum += get(m_Q, ii, jj) * dot_vec(vert_pos[ii], vert_pos[jj]); 
            }
        }
        *energy = 0.5f * stiffness * sum;
    }

    CONSTIF (compute_gradient)
    {
        for (uint ii = 0; ii < 4; ii++) 
        {
            for (uint jj = 0; jj < 4; jj++) 
            {
                force[ii] -= get(m_Q, ii, jj) * vert_pos[jj]; // -Qx
            }
            force[ii] = stiffness * force[ii];
        }
    }

    /// hessian
    //  0   1   2   3
    // t1   4   5   6
    // t2  t5   7   8
    // t3  t6  t8   9 
    CONSTIF (compute_hessian)
    {
        // uint idx = 0;
        // for (uint i = 0; i < 4; i++) 
        // {
        //     for (uint j = i; j < 4; j++)
        //     {
        //         hessian[idx] = stiffness * get(m_Q, j, i) * Identity3x3; 
        //         idx++;
        //     }
        // }
        hessian[0] = stiffness * get(m_Q, 0, 0) * Identity3x3;
        hessian[1] = stiffness * get(m_Q, 0, 1) * Identity3x3;
        hessian[2] = stiffness * get(m_Q, 0, 2) * Identity3x3;
        hessian[3] = stiffness * get(m_Q, 0, 3) * Identity3x3;
        hessian[4] = stiffness * get(m_Q, 1, 1) * Identity3x3;
        hessian[5] = stiffness * get(m_Q, 1, 2) * Identity3x3;
        hessian[6] = stiffness * get(m_Q, 1, 3) * Identity3x3;
        hessian[7] = stiffness * get(m_Q, 2, 2) * Identity3x3;
        hessian[8] = stiffness * get(m_Q, 2, 3) * Identity3x3;
        hessian[9] = stiffness * get(m_Q, 3, 3) * Identity3x3;
    }
}
template <bool compute_energy, bool compute_gradient, bool compute_hessian>
inline void stretch_mass_spring(
    THREAD float* energy, THREAD Float3* force, THREAD Float3x3* hessian,
    const THREAD Float3 vert_pos[2], 
    CREF(float) edge_rest_length, 
    const float stiffness_spring
)
{
    Float3 diff = vert_pos[1] - vert_pos[0];
    float l = max_scalar(length_vec(diff), Epsilon);
    float l0 = edge_rest_length;
    float C = l - l0;
    Float3 dir = diff / l;

    CONSTIF (compute_energy)
    {
        *energy = 0.5f * stiffness_spring * C * C;
    }

    CONSTIF (compute_gradient)
    {
        force[0] = stiffness_spring * dir * C;
        force[1] = -force[0];
    }

    CONSTIF (compute_hessian)
    {
        Float3x3 xxT = outer_product(diff, diff);
        float x_inv = 1.f / l;
        float x_squared_inv = x_inv * x_inv;
        Float3x3 He = stiffness_spring * x_squared_inv * xxT + stiffness_spring * max_scalar(1 - edge_rest_length * x_inv, 0.0f) * (Identity3x3 - x_squared_inv * xxT) ;
        /// hessian
        //  0   1 
        // t1   2  
        hessian[0] = He;
        hessian[1] = -1.f * He;
        hessian[2] = He;
    }

}


// Also Called As Isometric Bending
inline void solve_bending_quadratic_template(
	const uint eid, 
	const PTR(Float3) input_position, PTR(Float3) output_position, 
	PTR(Float3) sa_start_position, 
	PTR(ATOMIC_FLAG) sa_vert_mutex,
	PTR(float) lambda_bending, 
	PTR(float) sa_vert_mass_inv, PTR(Int4) sa_bending_edges, PTR(Float2) sa_bending_edge_adj_faces_area, 
	PTR(Float4x4) sa_bending_edge_Q, PTR(float) sa_bending_edge_rest_state_angle,
	const float stiffness_bending_quadratic, const float stiffness_bending_DBA, const float substep_dt, const bool use_atomic)
{
	Int4 edge = sa_bending_edges[eid];

	Float3 dx[4] = {Zero3, Zero3, Zero3, Zero3};

	// Constrains::solve_Bending_Isometric(eid, input_position,
	//     sa_vert_mass_inv.data(), sa_bending_edge_Q.data(), 
	//     edge, stiffness_bending, sub_step_dt, dx[0], dx[1], dx[2], dx[3]);
	
	const float vert_weight[4] = {
		sa_vert_mass_inv[edge[0]],
		sa_vert_mass_inv[edge[1]],
		sa_vert_mass_inv[edge[2]],
		sa_vert_mass_inv[edge[3]]
	};
	const Float3 vert_pos[4] = {
		input_position[edge[0]],
		input_position[edge[1]],
		input_position[edge[2]],
		input_position[edge[3]]
	};

	float lambda = lambda_bending[eid];
	Float3 edge_force[4] = {Zero3, Zero3, Zero3, Zero3};
	float energy = 0.f;
	Float4x4 m_Q = sa_bending_edge_Q[eid];

	core_bending_quadratic<true, true, false>(
		&energy, edge_force, nullptr, 
		vert_pos, m_Q, /*stiffness=*/ 1.0f);
	// energy *= stiffness_bending_quadratic;
	const float alpha_tilde = stiffness_bending_quadratic < Epsilon ? 1e12 :  1.f / (stiffness_bending_quadratic * substep_dt * substep_dt);
	
	const float delta_lambda = Core::get_delta_xpbd_4(
		dx[0], dx[1], dx[2], dx[3],
		-edge_force[0], -edge_force[1], -edge_force[2], -edge_force[3],
		vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3],
		energy, lambda, alpha_tilde);

	// if (vert_weight[0] == 0 || vert_weight[1] == 0 || vert_weight[2] == 0 || vert_weight[3] == 0)
	// {
	// 	dx[0] = Zero3;
	// 	dx[1] = Zero3;
	// 	dx[2] = Zero3;
	// 	dx[3] = Zero3;
	// }

	lambda_bending[eid] = lambda + delta_lambda;

	Core::assemble_to_position(output_position, sa_vert_mutex, edge, dx, use_atomic);
};

inline float fast_acos(const float dot) { return (-0.6981317 * dot * dot - 0.8726646) * dot + 1.570796; }

// Muller 2007
inline void solve_bending_DAB_template(
	const uint eid, const PTR(Float3) input_position, PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_start_position, 
	PTR(ATOMIC_FLAG) sa_vert_mutex, PTR(float) lambda_bending, 
	PTR(float) sa_vert_mass_inv, PTR(Int4) sa_bending_edges, PTR(Float2) sa_bending_edge_adj_faces_area, 
	PTR(Float4x4) sa_bending_edge_Q, PTR(float) sa_bending_edge_rest_state_angle,
	const float stiffness_bending_quadratic, const float stiffness_bending_DAB, const float substep_dt, const bool use_atomic
)
{
	// Int4 edge = sa_bending_edges[eid];

	// Float3 dx[4] = {Zero3, Zero3, Zero3, Zero3};
	// Float3 force[4] = {Zero3, Zero3, Zero3, Zero3};
	
	// const float vert_weight[4] = {
	// 	sa_vert_mass_inv[edge[0]],
	// 	sa_vert_mass_inv[edge[1]],
	// 	sa_vert_mass_inv[edge[2]],
	// 	sa_vert_mass_inv[edge[3]]
	// };

	// const Float3 vert_pos[4] = {
	// 	input_position[edge[0]],
	// 	input_position[edge[1]],
	// 	input_position[edge[2]],
	// 	input_position[edge[3]]
	// };

	Int4 edge = sa_bending_edges[eid];

	const float eps = 1e-6;
	const float ke = stiffness_bending_DAB;
	const float kd = 0.0;

	const uint i = edge[0];
	const uint j = edge[1];
	const uint k = edge[2];
	const uint l = edge[3];

	const Float3 x1 = input_position[i];
	const Float3 x2 = input_position[j];
	const Float3 x3 = input_position[k];
	const Float3 x4 = input_position[l];

	// const Float3 v1 = (x1 - sa_start_position[i]) / substep_dt;
	// const Float3 v2 = (x2 - sa_start_position[j]) / substep_dt;
	// const Float3 v3 = (x3 - sa_start_position[k]) / substep_dt;
	// const Float3 v4 = (x4 - sa_start_position[l]) / substep_dt;

	const float w1 = sa_vert_mass_inv[i];
	const float w2 = sa_vert_mass_inv[j];
	const float w3 = sa_vert_mass_inv[k];
	const float w4 = sa_vert_mass_inv[l];

	const Float3 p2 = x2 - x1;
	const Float3 p3 = x3 - x1;
	const Float3 p4 = x4 - x1;

	//
	// Refer to genesis: https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/engine/solvers/pbd_solver.py
	//
	Float3 n1 = cross_vec(p2, p3); // # normal to face 1
	Float3 n2 = cross_vec(p2, p4); // # normal to face 2
	float l23 = length_vec(n1);
	float l24 = length_vec(n2);

	if (l23 < eps or l24 < eps) return;

	// const float rest_angle = sa_bending_edge_rest_state_angle[eid]; 
	// const float rest_angle = 0.0; 

	n1 = n1 / l23;
	n2 = n2 / l24;

	//
	// C = theta - theta_0
	// dC = -1/sqrt(1-d^2) * q
	// delta_lambda = (-C - alpha_tilde * lambda) / (sum_i^4(w_i dC^2 ) + alpha_tilde)
	//              = (-C - alpha_tilde * lambda) / (sum_i^4(w_i q^2 / (1-d^2) ) + alpha_tilde) // multiply by (1-d^2)
	//              = (-C - alpha_tilde * lambda) * (1-d^2) / (sum_i^4(w_i q^2 ) + alpha_tilde * (1-d^2))
	//
	// dx = M^-1 * dC * delta_lambda
	//    = w * dC * (-C - alpha_tilde * lambda) * (1-d^2) / (sum_i^4(w_i q^2 ) + alpha_tilde * (1-d^2))
	//    = w * dC * (-C - alpha_tilde * lambda) * (1-d^2) / (sum_i^4(w_i q^2 ) + alpha_tilde * (1-d^2))
	//    = w * q * (-(1-d^2)/sqrt(1-d^2)) * (-C - alpha_tilde * lambda) / (sum_i^4(w_i q^2 ) + alpha_tilde * (1-d^2))
	//    = w * q * (-sqrt(1-d^2)) * (-C - alpha_tilde * lambda) / (sum_i^4(w_i q^2 ) + alpha_tilde * (1-d^2))
	//    = w * q * sqrt(1-d^2) * (C + alpha_tilde * lambda) / (sum_i^4(w_i q^2 ) + alpha_tilde * (1-d^2))
	//
	//    (For PBD instead of XPBD)
	//    = w * q * sqrt(1-d^2) * C / (sum_i^4(w_i q^2 ))
	//    = Position Based Dynamics A.14
	//

	const float d = clamp_scalar(dot_vec(n1, n2), -1.0f, 1.0f); // Remember to clamp!!!! Otherwise `acos(cos_theta)` may be NaN
	float C = acos_scalar(d) - Pi; // - acos_scalar(-1.0)
	float sin_theta_sqr = 1 - d * d; // sin^2(theta) = 1 - cos^2(theta)
	float sin_theta = sqrt_scalar(sin_theta_sqr); // d_cos(theta) / d_x = sin(theta) = sqrt(1 - cos^2(theta))
	
	// if (sin_theta < Epsilon) return;

	Float3 q3 =  (cross_vec(p2, n2) + cross_vec(n1, p2) * d) / l23; //  # eq. (25)
	Float3 q4 =  (cross_vec(p2, n1) + cross_vec(n2, p2) * d) / l24; //  # eq. (26)
	Float3 q2 = -(cross_vec(p3, n2) + cross_vec(n1, p3) * d) / l23
				-(cross_vec(p4, n1) + cross_vec(n2, p4) * d) / l24; //  # eq. (27)
	Float3 q1 = -q2 - q3 - q4;

	float alpha_tilde = ke < Epsilon ? 1e12 : 1.0 / (ke * substep_dt * substep_dt);
	// float alpha_tilde = 1e9;
	
	// # eq. (29)
	float sum_wq = (
		  w1 * length_squared_vec(q1)
		+ w2 * length_squared_vec(q2)
		+ w3 * length_squared_vec(q3)
		+ w4 * length_squared_vec(q4)
	);
	// float denom = sum_wq + alpha_tilde * sin_theta_sqr;
	
	// # Note strict inequality for damping -- 0 damping is ok
	if (sum_wq <= 0.0 or ke <= 0.0 or kd < 0.0) return;

	float orig_lambda = lambda_bending[eid];
	// float grad_dot_v = substep_dt * (dot_vec(q1, v1) + dot_vec(q2, v2) + dot_vec(q3, v3) + dot_vec(q4, v4));
	// float dlambda = -1.0 * (c + alpha * orig_lambda + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha);
	
	// float dlambda = (-C - alpha_tilde * orig_lambda) / (sum_wq + alpha_tilde);
	float dlambda = (
		sin_theta * (-C - alpha_tilde * orig_lambda)
		/ (sum_wq + alpha_tilde * sin_theta_sqr)
	);

	// float multiplier = -1.0f / sqrt_scalar(1 - d * d);
	Float3 delta0 = w1 * dlambda * q1;
	Float3 delta1 = w2 * dlambda * q2;
	Float3 delta2 = w3 * dlambda * q3;
	Float3 delta3 = w4 * dlambda * q4;

	lambda_bending[eid] = orig_lambda + sin_theta * dlambda;
	// lambda_bending[eid] = orig_lambda + dlambda;
	sa_iter_position[i] += delta0;
	sa_iter_position[j] += delta1;
	sa_iter_position[k] += delta2;
	sa_iter_position[l] += delta3;   
	
	
};


//
// Bending Constraint Copy From Unreal Engine
//         Engine/Source/Runtime/Experimental/Chaos/Private/Chaos/XPBDBendingElement.isph
//
using FVector3f = Float3;
CONSTEXPR float FLOAT_SMALL_NUMBER = 1e-8f;

static inline FVector3f SafeDivide(CREF(Float3) Numerator, const float Denominator)
{
	if (Denominator <= FLOAT_SMALL_NUMBER) return Zero3;
	else return Numerator / Denominator;
}
// static inline FVector3f VectorGetSafeNormal(CREF(Float3) vec)
// {
// 	const float squaredLength = length_squared_vec(vec);
// 	if (squaredLength <= FLOAT_SMALL_NUMBER) return Zero3;
// 	else return vec / sqrt_scalar(squaredLength);
// }
static inline FVector3f VectorGetSafeNormal(CREF(Float3) Vector)
{
	const float SquareSum = length_squared_vec(Vector);

	// Not sure if it's safe to add tolerance in there. Might introduce too many errors
	if (SquareSum == 1.f)
	{
		return Vector;
	}
	else if (SquareSum < FLOAT_SMALL_NUMBER)
	{
		return Zero3;
	}
	const float Scale = rsqrt_scalar(SquareSum);
	return Vector * Scale;
}

inline float CalcGradientsAndAngle(CREF(Float3) P1, CREF(Float3) P2, CREF(Float3) P3, CREF(Float3) P4, 
	TREF(Float3) Grad1, TREF(Float3) Grad2, TREF(Float3) Grad3, TREF(Float3) Grad4)
{
	const FVector3f SharedEdgeNormalized = VectorGetSafeNormal(P2 - P1);

	const FVector3f P13CrossP23 = cross_vec(P1 - P3, P2 - P3);
	const float Normal1Len = length_vec(P13CrossP23);
	const FVector3f Normal1 = SafeDivide(P13CrossP23, Normal1Len);

	const FVector3f P24CrossP14 = cross_vec(P2 - P4, P1 - P4);
	const float Normal2Len = length_vec(P24CrossP14);
	const FVector3f Normal2 = SafeDivide(P24CrossP14, Normal2Len);

	const FVector3f N2CrossN1 = cross_vec(Normal2, Normal1);

	const float CosPhi = clamp_scalar(dot_vec(Normal1, Normal2), (float)(-1), (float)(1));
	const float SinPhi = clamp_scalar(dot_vec(N2CrossN1, SharedEdgeNormalized), (float)(-1), (float)(1));
	// const float SinPhi = clamp_scalar(dot_vec(N2CrossN1, SharedEdgeNormalized), 1e-8, (float)(1));

	const float Angle = atan2_scalar(SinPhi, CosPhi); // if CosPhi == 0, atan(sin/cos) -> nan, so we use safe atan2
	// const float Angle = acos_scalar(CosPhi);

	const FVector3f DPhiDN1_OverNormal1Len = SafeDivide(CosPhi * cross_vec(SharedEdgeNormalized, Normal2) - SinPhi * Normal2, Normal1Len);
	const FVector3f DPhiDN2_OverNormal2Len = SafeDivide(CosPhi * cross_vec(Normal1, SharedEdgeNormalized) - SinPhi * Normal1, Normal2Len);

	const FVector3f DPhiDP13 = cross_vec(P2 - P3, DPhiDN1_OverNormal1Len);
	const FVector3f DPhiDP23 = cross_vec(DPhiDN1_OverNormal1Len, P1 - P3);
	const FVector3f DPhiDP24 = cross_vec(P1 - P4, DPhiDN2_OverNormal2Len);
	const FVector3f DPhiDP14 = cross_vec(DPhiDN2_OverNormal2Len, P2 - P4);

	// 梯度分配到四个点
	Grad1 = DPhiDP13 + DPhiDP14;
	Grad2 = DPhiDP23 + DPhiDP24;
	Grad3 = -1 * DPhiDP13 - DPhiDP23;
	Grad4 = -1 * DPhiDP14 - DPhiDP24;

	return Angle;
}

inline void solve_bending_DAB_template_v2(
	const uint eid, const PTR(Float3) input_position, PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_start_position, 
	PTR(ATOMIC_FLAG) sa_vert_mutex, PTR(float) lambda_bending, 
	PTR(float) sa_vert_mass_inv, PTR(Int4) sa_bending_edges, PTR(Float2) sa_bending_edge_adj_faces_area, 
	PTR(Float4x4) sa_bending_edge_Q, PTR(float) sa_bending_edge_rest_state_angle,
	const float stiffness_bending_quadratic, const float stiffness_bending_DAB, const float substep_dt, const bool use_atomic
)
{
	Int4 edge = sa_bending_edges[eid];

    // const float eps = 1e-6;
    const float ke = stiffness_bending_DAB; // 弯曲刚度
    const float kd = 0.0f; // 阻尼系数（可以调整）

    const uint i = edge[0];
    const uint j = edge[1];
    const uint k = edge[2];
    const uint l = edge[3];

    const Float3 x1 = input_position[i];
    const Float3 x2 = input_position[j];
    const Float3 x3 = input_position[k];
    const Float3 x4 = input_position[l];

    // const Float3 v1 = (x1 - sa_start_position[i]) / substep_dt;
    // const Float3 v2 = (x2 - sa_start_position[j]) / substep_dt;
    // const Float3 v3 = (x3 - sa_start_position[k]) / substep_dt;
    // const Float3 v4 = (x4 - sa_start_position[l]) / substep_dt;

    const float w1 = sa_vert_mass_inv[i];
    const float w2 = sa_vert_mass_inv[j];
    const float w3 = sa_vert_mass_inv[k];
    const float w4 = sa_vert_mass_inv[l];

    Float3 Grad1, Grad2, Grad3, Grad4;
	// if (
	// 	length_squared_vec(x1 - x2) < 1e-8
	// 	|| is_nan_vec(x1) || is_nan_vec(x2) || is_nan_vec(x3) || is_nan_vec(x4)
	// ) return;
	
    // float Angle = CalcGradientsAndAngle(x1, x2, x3, x4, Grad1, Grad2, Grad3, Grad4);
	float Angle = 0.0f;
	{
		const FVector3f SharedEdgeNormalized = VectorGetSafeNormal(x2 - x1);

		const FVector3f P13CrossP23 = cross_vec(x1 - x3, x2 - x3);
		const float Normal1Len = length_vec(P13CrossP23);
		const FVector3f Normal1 = SafeDivide(P13CrossP23, Normal1Len);

		const FVector3f P24CrossP14 = cross_vec(x2 - x4, x1 - x4);
		const float Normal2Len = length_vec(P24CrossP14);
		const FVector3f Normal2 = SafeDivide(P24CrossP14, Normal2Len);

		const FVector3f N2CrossN1 = cross_vec(Normal2, Normal1);

		const float CosPhi = clamp_scalar(dot_vec(Normal1, Normal2), (float)(-0.999999), (float)(0.999999));
		const float SinPhi = clamp_scalar(dot_vec(N2CrossN1, SharedEdgeNormalized), (float)(-0.999999), (float)(0.999999));
		// const float SinPhi = clamp_scalar(dot_vec(N2CrossN1, SharedEdgeNormalized), 1e-8, (float)(1));

		if (Normal1Len < FLOAT_SMALL_NUMBER || Normal2Len < FLOAT_SMALL_NUMBER || abs_scalar(CosPhi) < FLOAT_SMALL_NUMBER) return;

		Angle = atan2_scalar(SinPhi, CosPhi); // if CosPhi == 0, atan(sin/cos) -> nan, so we use safe atan2
		// const float Angle = acos_scalar(CosPhi);

		const FVector3f DPhiDN1_OverNormal1Len = SafeDivide(CosPhi * cross_vec(SharedEdgeNormalized, Normal2) - SinPhi * Normal2, Normal1Len);
		const FVector3f DPhiDN2_OverNormal2Len = SafeDivide(CosPhi * cross_vec(Normal1, SharedEdgeNormalized) - SinPhi * Normal1, Normal2Len);

		const FVector3f DPhiDP13 = cross_vec(x2 - x3, DPhiDN1_OverNormal1Len);
		const FVector3f DPhiDP23 = cross_vec(DPhiDN1_OverNormal1Len, x1 - x3);
		const FVector3f DPhiDP24 = cross_vec(x1 - x4, DPhiDN2_OverNormal2Len);
		const FVector3f DPhiDP14 = cross_vec(DPhiDN2_OverNormal2Len, x2 - x4);

		Grad1 = DPhiDP13 + DPhiDP14;
		Grad2 = DPhiDP23 + DPhiDP24;
		Grad3 = -1 * DPhiDP13 - DPhiDP23;
		Grad4 = -1 * DPhiDP14 - DPhiDP24;
	}
    const float RestAngle = sa_bending_edge_rest_state_angle[eid];

    const float CombinedInvMass = w1 + w2 + w3 + w4;
    const float Damping = 2.f * kd * sqrt_scalar(ke / CombinedInvMass);
    const float BetaDt = Damping * substep_dt;

    const float AlphaInv = ke * substep_dt * substep_dt;
    float ElasticTerm = AlphaInv * (Angle - RestAngle);
    float DampingTerm = 0.0f;
	if (kd != 0.0f) 
	{
		const Float3 V1TimesDt = x1 - sa_start_position[i];
		const Float3 V2TimesDt = x2 - sa_start_position[j];
		const Float3 V3TimesDt = x3 - sa_start_position[k];
		const Float3 V4TimesDt = x4 - sa_start_position[l];
		DampingTerm = BetaDt * (dot_vec(V1TimesDt, Grad1) + dot_vec(V2TimesDt, Grad2) + dot_vec(V3TimesDt, Grad3) + dot_vec(V4TimesDt, Grad4));
	}

	{
		// const FVector3f SharedEdgeNormalized = VectorGetSafeNormal(x2 - x1);
		// const FVector3f P13CrossP23 = cross_vec(x1 - x3, x2 - x3);
		// const float Normal1Len = length_vec(P13CrossP23);
		// const FVector3f Normal1 = SafeDivide(P13CrossP23, Normal1Len);
		// const FVector3f P24CrossP14 = cross_vec(x2 - x4, x1 - x4);
		// const float Normal2Len = length_vec(P24CrossP14);
		// const FVector3f Normal2 = SafeDivide(P24CrossP14, Normal2Len);
		// const FVector3f N2CrossN1 = cross_vec(Normal2, Normal1);
		// const float CosPhi = clamp_scalar(dot_vec(Normal1, Normal2), (float)(-1), (float)(1));
		// const float SinPhi = clamp_scalar(dot_vec(N2CrossN1, SharedEdgeNormalized), (float)(-1), (float)(1));
		// if (is_nan_scalar(ElasticTerm) || is_nan_scalar(DampingTerm))
		// {
		// 	sa_bending_edge_Q[eid] = transpose_mat(make<Float4x4>(
		// 		makeFloat4(1.f, RestAngle, ElasticTerm, DampingTerm),
		// 		makeFloat4(Angle, CombinedInvMass, Damping, BetaDt),
		// 		makeFloat4(CosPhi, SinPhi, CosPhi/SinPhi, AlphaInv),
		// 		makeFloat4(Normal1Len, Normal2Len, atan_scalar(CosPhi/SinPhi), acos_scalar(CosPhi))));
		// }
		// else sa_bending_edge_Q[eid] = make<Float4x4>(0.f);
	}
	
    const float Denom = (AlphaInv + BetaDt) * (
        w1 * length_squared_vec(Grad1) +
        w2 * length_squared_vec(Grad2) +
        w3 * length_squared_vec(Grad3) +
        w4 * length_squared_vec(Grad4)
    ) + 1.f;

    float OrigLambda = lambda_bending[eid];
    float DLambda = (-ElasticTerm - DampingTerm - OrigLambda) / Denom;

    Float3 Delta1 = DLambda * w1 * Grad1;
    Float3 Delta2 = DLambda * w2 * Grad2;
    Float3 Delta3 = DLambda * w3 * Grad3;
    Float3 Delta4 = DLambda * w4 * Grad4;

    lambda_bending[eid] = OrigLambda + DLambda;

    sa_iter_position[i] += Delta1;
    sa_iter_position[j] += Delta2;
    sa_iter_position[k] += Delta3;
    sa_iter_position[l] += Delta4;
	
};


inline void solve_tetrahedral_fem_NeoHookean_template(
	const uint tet_id, const PTR(Float3) input_position, PTR(Float3) output_position, PTR(Float3) start_position, 
	PTR(ATOMIC_FLAG) sa_vert_mutex, PTR(float) lambda_tet_neohookean_hydrostatic_term, PTR(float) lambda_tet_neohookean_deviatoric_term, 
	PTR(float) sa_vert_mass_inv, PTR(Int4) sa_tets, PTR(float) sa_tet_volumn, PTR(Float3x3) sa_Dm_inv, 
	const float m_first_lame, const float m_second_lame, const float substep_dt, const bool use_atomic
	)
{
	const Int4 tet = sa_tets[tet_id];
	
	Float3 vert_pos[4] = {
		input_position[tet[0]],
		input_position[tet[1]],
		input_position[tet[2]],
		input_position[tet[3]]
	};
	const float vert_weight[4] = {
		sa_vert_mass_inv[tet[0]],
		sa_vert_mass_inv[tet[1]],
		sa_vert_mass_inv[tet[2]],
		sa_vert_mass_inv[tet[3]]
	};
	const float tet_volumn = sa_tet_volumn[tet_id];
	const Float3x3 Dm_inv = sa_Dm_inv[tet_id];
	const Float3x3 Dm_inv_T = transpose_mat(Dm_inv);

	const float lambda_alpha = lambda_tet_neohookean_hydrostatic_term[tet_id]; 
	const float lambda_mu  = lambda_tet_neohookean_deviatoric_term[tet_id];
	// const float alpha_tilde_hydrostatic = 0;
	// const float alpha_tilde_deviatoric  = 0;
	const float alpha_tilde_alpha = m_first_lame  < Epsilon ? 1e9 : 1.f / (m_first_lame * tet_volumn * substep_dt * substep_dt);
	const float alpha_tilde_mu  = m_second_lame < Epsilon ? 1e9 : 1.f / (m_second_lame * tet_volumn * substep_dt * substep_dt);
	
	// gamma_tilde = alpha_tilde * beta_tilde / h = 1/(stiff*h^2)*(h^2*k_damping)/h=k_damping/(stiff*h)
	// const float k_damping = 0.1f;
	// const float gamma_tilde_alpha = k_damping / (substep_dt * m_first_lame * tet_volumn);
	// const float gamma_tilde_mu = k_damping / (substep_dt * m_second_lame * tet_volumn);
	// const Float3 start_pos[4] = {
	// 	start_position[tet[0]],
	// 	start_position[tet[1]],
	// 	start_position[tet[2]],
	// 	start_position[tet[3]]
	// };

	Float3 dx[4] = {Zero3, Zero3, Zero3, Zero3};

	//
	// Constraint Formulation of Stable Neo-Hookean
 	//
	/*
	{
		// Volumn Term (Hydrostatic)
		{
			const Float3x3 Ds = makeFloat3x3(
				vert_pos[1] - vert_pos[0], 
				vert_pos[2] - vert_pos[0], 
				vert_pos[3] - vert_pos[0]);
			const Float3x3 F = Ds * Dm_inv;

			const float J = determinant_mat(F);
			// if (J > Epsilon)
			{
				const float C = J - 1.0f;

				const THREAD Float3& f1 = get(F, 0);
				const THREAD Float3& f2 = get(F, 1);
				const THREAD Float3& f3 = get(F, 2);
				const Float3x3 dJdF = makeFloat3x3( cross_vec(f2, f3), cross_vec(f3, f1), cross_vec(f1, f2) );
				const Float3x3 gradient_234 = dJdF * Dm_inv_T;

				const THREAD Float3& g2 = get(gradient_234, 0);
				const THREAD Float3& g3 = get(gradient_234, 1);
				const THREAD Float3& g4 = get(gradient_234, 2);
				const Float3 g1 = -(g2 + g3 + g4); 

				const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4(
					dx[0], dx[1], dx[2], dx[3], 
					g1, g2, g3, g4,
					vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
					C, lambda_deviatoric, alpha_tilde_hydrostatic);

				lambda_tet_neohookean_deviatoric_term[tet_id] += delta_lambda; 
			}	
		}
		{
			// vert_pos[0] += dx[0]; 
			// vert_pos[1] += dx[1]; 
			// vert_pos[2] += dx[2]; 
			// vert_pos[3] += dx[3]; 
		}
		// Distortion Term (Deviatoric)
		{
			const Float3x3 Ds = makeFloat3x3(
				vert_pos[1] - vert_pos[0], 
				vert_pos[2] - vert_pos[0], 
				vert_pos[3] - vert_pos[0]);
			const Float3x3 F = Ds * Dm_inv;

			const THREAD Float3& f1 = get(F, 0);
			const THREAD Float3& f2 = get(F, 1);
			const THREAD Float3& f3 = get(F, 2);

			const float tr = length_squared_vec(f1) + length_squared_vec(f2) + length_squared_vec(f3);
			// const float J = determinant_mat(F); // det
			// if (J > Epsilon)
			{
				const float C = tr - 3.0f;
				const Float3x3 gradient_234 = 2.0f * F * dFdx;

				const THREAD Float3& g2 = get(gradient_234, 0);
				const THREAD Float3& g3 = get(gradient_234, 1);
				const THREAD Float3& g4 = get(gradient_234, 2);
				const Float3 g1 = -(g2 + g3 + g4); 

				const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4(
					dx[0], dx[1], dx[2], dx[3], 
					g1, g2, g3, g4,
					vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
					C, lambda_hydrostatic, alpha_tilde_deviatoric);

				lambda_tet_neohookean_hydrostatic_term[tet_id] += delta_lambda; 
			}
		}
		
	}
	*/

	//
	// From : A Constraint-based Formulation of Stable Neo-Hookean Materials
	//
	{
		// const Float3 h_mult_velocity[4] = {
		// 	(vert_pos[0] - start_pos[0]),
		// 	(vert_pos[1] - start_pos[1]),
		// 	(vert_pos[2] - start_pos[2]),
		// 	(vert_pos[3] - start_pos[3])
		// };


		// Deviatoric Term
		{
			const Float3x3 Ds = makeFloat3x3(
				vert_pos[1] - vert_pos[0], 
				vert_pos[2] - vert_pos[0], 
				vert_pos[3] - vert_pos[0]);
			
			// const Float3 h_mult_velocity[4] = {
			// 	(vert_pos[0] - start_pos[0]),
			// 	(vert_pos[1] - start_pos[1]),
			// 	(vert_pos[2] - start_pos[2]),
			// 	(vert_pos[3] - start_pos[3])
			// };

			const Float3x3 F = Ds * Dm_inv;
			const THREAD Float3& f1 = get(F, 0);
			const THREAD Float3& f2 = get(F, 1);
			const THREAD Float3& f3 = get(F, 2);
			
			const float tr = length_squared_vec(f1) + length_squared_vec(f2) + length_squared_vec(f3);
			
			// const float why_delete_this_term = sqrt_scalar(3.0f);
			// const float rs = sqrt_scalar(tr - 3.0f);
			// const float rs_inv = rs == 0.0f ? 0.0f : 1.f / rs;
			// const float C_D = rs;
			// const Float3x3 gradient_D = rs_inv * F * Dm_inv_T;

			const float C_D = tr - 3.0f;
			const Float3x3 gradient_D = 2.0f * F * Dm_inv_T;
			const THREAD Float3& g2 = get(gradient_D, 0);
			const THREAD Float3& g3 = get(gradient_D, 1);
			const THREAD Float3& g4 = get(gradient_D, 2);
			// const float inv_rs = 1.0f / rs;
			// const Float3 g2 = inv_rs * f1 * Dm_inv_T;
			// const Float3 g3 = inv_rs * f2 * Dm_inv_T;
			// const Float3 g4 = inv_rs * f3 * Dm_inv_T;
			const Float3 g1 = -(g2 + g3 + g4); 

			// const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4_with_damping(
			// 	dx[0], dx[1], dx[2], dx[3], 
			// 	g1, g2, g3, g4,
			// 	h_mult_velocity[0], h_mult_velocity[1], h_mult_velocity[2], h_mult_velocity[3],
			// 	vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
			// 	C_D, lambda_mu, alpha_tilde_mu, gamma_tilde_mu);
			const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4(
				dx[0], dx[1], dx[2], dx[3], 
				g1, g2, g3, g4,
				vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
				C_D, lambda_mu, alpha_tilde_mu);

			lambda_tet_neohookean_deviatoric_term[tet_id] += delta_lambda; 
		}
		// {
		// 	vert_pos[0] += dx[0]; 
		// 	vert_pos[1] += dx[1]; 
		// 	vert_pos[2] += dx[2]; 
		// 	vert_pos[3] += dx[3]; 
		// }
		// Hydrostatic Term / Volumn Term
		{
			const Float3x3 Ds = makeFloat3x3(
				vert_pos[1] - vert_pos[0], 
				vert_pos[2] - vert_pos[0], 
				vert_pos[3] - vert_pos[0]);
			
			
			const Float3x3 F = Ds * Dm_inv;
			const THREAD Float3& f1 = get(F, 0);
			const THREAD Float3& f2 = get(F, 1);
			const THREAD Float3& f3 = get(F, 2);
			
			const float C_H = determinant_mat(F) - 1.0f;// - m_second_lame / m_first_lame;

			const Float3x3 gradient_H = makeFloat3x3( cross_vec(f2, f3), cross_vec(f3, f1), cross_vec(f1, f2) ) * Dm_inv_T;
			const THREAD Float3& g2 = get(gradient_H, 0);
			const THREAD Float3& g3 = get(gradient_H, 1);
			const THREAD Float3& g4 = get(gradient_H, 2);
			// const Float3 g2 = cross_vec(f2, f3) * Dm_inv_T;
			// const Float3 g3 = cross_vec(f3, f1) * Dm_inv_T;
			// const Float3 g4 = cross_vec(f1, f2) * Dm_inv_T;
			const Float3 g1 = -(g2 + g3 + g4); 

			// const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4_with_damping(
			// 	dx[0], dx[1], dx[2], dx[3], 
			// 	g1, g2, g3, g4,
			// 	h_mult_velocity[0], h_mult_velocity[1], h_mult_velocity[2], h_mult_velocity[3],
			// 	vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
			// 	C_H, lambda_alpha, alpha_tilde_alpha, gamma_tilde_alpha);
			const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4(
				dx[0], dx[1], dx[2], dx[3], 
				g1, g2, g3, g4,
				vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
				C_H, lambda_alpha, alpha_tilde_alpha);

			lambda_tet_neohookean_hydrostatic_term[tet_id] += delta_lambda; 
		}
	}
	Constrains::Core::assemble_to_position(output_position, sa_vert_mutex, tet, dx, use_atomic);  
		
};

inline void solve_tetrahedral_fem_StVK_template(
	const uint tet_id, const PTR(Float3) input_position, PTR(Float3) output_position, PTR(Float3) sa_start_position,
	PTR(ATOMIC_FLAG) sa_vert_mutex, PTR(float) lambda_tet_neohookean_hydrostatic_term, PTR(float) lambda_tet_neohookean_deviatoric_term, 
	PTR(float) sa_vert_mass_inv, PTR(Int4) sa_tets, PTR(float) sa_tet_volumn, PTR(Float3x3) sa_Dm_inv, 
	const float m_first_lame, const float m_second_lame, const float substep_dt, const bool use_atomic
	)
{
	const Int4 tet = sa_tets[tet_id];
	Float3 vert_pos[4] = {
		input_position[tet[0]],
		input_position[tet[1]],
		input_position[tet[2]],
		input_position[tet[3]]
	};
	const float vert_weight[4] = {
		sa_vert_mass_inv[tet[0]],
		sa_vert_mass_inv[tet[1]],
		sa_vert_mass_inv[tet[2]],
		sa_vert_mass_inv[tet[3]]
	};
	const float tet_volumn = sa_tet_volumn[tet_id];
	const Float3x3 Dm_inv = sa_Dm_inv[tet_id];
	const Float3x3 Dm_inv_T = transpose_mat(Dm_inv);
	// const THREAD Float3x3& dFdx = Dm_inv_T;

	const float lambda_hydrostatic = lambda_tet_neohookean_hydrostatic_term[tet_id]; 
	const float lambda_deviatoric  = lambda_tet_neohookean_deviatoric_term[tet_id];
	const float alpha_tilde_lambda = m_first_lame  < Epsilon ? 1e12 : 1.f / (m_first_lame * tet_volumn * substep_dt * substep_dt);
	const float alpha_tilde_mu  = m_second_lame < Epsilon ? 1e12 : 1.f / (m_second_lame * tet_volumn * substep_dt * substep_dt);
	
	Float3 dx[4] = {Zero3, Zero3, Zero3, Zero3};

	//
	// From : Position-based simulation of continuous materials
	//
	
	{
		// Volumn Term
		{
			const Float3x3 Ds = makeFloat3x3(
				vert_pos[1] - vert_pos[0], 
				vert_pos[2] - vert_pos[0], 
				vert_pos[3] - vert_pos[0]);

			const Float3x3 F = Ds * Dm_inv;

			const Float3x3 epsilon = 0.5f * (transpose_mat(F) * F - Identity3x3);
			const float trace_eps = trace_mat(epsilon);

			const float energy = 0.5 * m_first_lame * tet_volumn * trace_eps * trace_eps;
			const Float3x3 P = m_first_lame * trace_eps * F;
			const Float3x3 gradient_234 = tet_volumn * P * Dm_inv_T;

			const THREAD Float3& g2 = get(gradient_234, 0);
			const THREAD Float3& g3 = get(gradient_234, 1);
			const THREAD Float3& g4 = get(gradient_234, 2);
			const Float3 g1 = -(g2 + g3 + g4); 

			const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4(
				dx[0], dx[1], dx[2], dx[3], 
				g1, g2, g3, g4,
				vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
				energy, lambda_deviatoric, alpha_tilde_lambda);

			lambda_tet_neohookean_deviatoric_term[tet_id] += delta_lambda; 

			{
				vert_pos[0] += dx[0]; 
				vert_pos[1] += dx[1]; 
				vert_pos[2] += dx[2]; 
				vert_pos[3] += dx[3]; 
			}
		}
	
		// Shearing Term
		{
			const Float3x3 Ds = makeFloat3x3(
				vert_pos[1] - vert_pos[0], 
				vert_pos[2] - vert_pos[0], 
				vert_pos[3] - vert_pos[0]);

			const Float3x3 F = Ds * Dm_inv;

			const Float3x3 epsilon = 0.5f * (transpose_mat(F) * F - Identity3x3);
			const float trace_squared_eps = frobenius_squared_norm_mat(epsilon);

			const float energy = m_second_lame * tet_volumn * trace_squared_eps;
			const Float3x3 P = 2.0f * m_second_lame * F * epsilon;
			const Float3x3 gradient_234 = tet_volumn * P * Dm_inv_T;

			const THREAD Float3& g2 = get(gradient_234, 0);
			const THREAD Float3& g3 = get(gradient_234, 1);
			const THREAD Float3& g4 = get(gradient_234, 2);
			const Float3 g1 = -(g2 + g3 + g4); 

			const float delta_lambda = Constrains::Core::get_additional_delta_x_xpbd_4(
				dx[0], dx[1], dx[2], dx[3], 
				g1, g2, g3, g4,
				vert_weight[0], vert_weight[1], vert_weight[2], vert_weight[3], 
				energy, lambda_hydrostatic, alpha_tilde_mu);

			lambda_tet_neohookean_hydrostatic_term[tet_id] += delta_lambda; 
		}
	}
	
	Constrains::Core::assemble_to_position(output_position, sa_vert_mutex, tet, dx, use_atomic); 
		
};





inline Float2 solve_cloth_balloon_get_volumn_pass(
	const uint fid, const PTR(Float3) input_position, PTR(Float3) output_position, 
	PTR(ATOMIC_FLAG) sa_vert_mutex, 
	PTR(float) sa_vert_mass_inv, PTR(Int3) sa_faces, 
	const float balloon_scale_rate, const float stiffness_pressure, const float substep_dt, const bool use_atomic
)
{
	const Int3 face = sa_faces[fid];
	const float vert_weight[3] = {
		sa_vert_mass_inv[face[0]],
		sa_vert_mass_inv[face[1]],
		sa_vert_mass_inv[face[2]]
	};
	const Float3 vert_pos[3] = {
		input_position[face[0]],
		input_position[face[1]],
		input_position[face[2]]
	};
	const Float3 gradient0 = cross_vec(vert_pos[1], vert_pos[2]);
	const Float3 gradient1 = cross_vec(vert_pos[2], vert_pos[0]);
	const Float3 gradient2 = cross_vec(vert_pos[0], vert_pos[1]);

	// vert_gradient_mult_weight_cloth_balloon[face[0]] += vert_weight[0] * gradient0;
	// vert_gradient_mult_weight_cloth_balloon[face[1]] += vert_weight[1] * gradient1;
	// vert_gradient_mult_weight_cloth_balloon[face[2]] += vert_weight[2] * gradient2;

	const float volumn = dot_vec(cross_vec(vert_pos[0], vert_pos[1]), vert_pos[2]);
	const float sum_w_sng = 
		vert_weight[0] * length_squared_vec(gradient0) + 
		vert_weight[1] * length_squared_vec(gradient1) + 
		vert_weight[2] * length_squared_vec(gradient2); // Sum of Weighted Squared Norm Of Gradient

	return makeFloat2(volumn, sum_w_sng);
}

//
// lambda_cloth_balloon :
//    0 : lambda
//	  1 : delta lambda
//	  2 : reduce
//
inline void solve_cloth_balloon_save_global_volumn(
	const uint cloth_idx, const Float2 reduce_result, 
	PTR(float) sa_cloth_volumn, PTR(float) lambda_cloth_balloon)
{
#ifdef METAL_CODE
	atomic_add(sa_cloth_volumn[cloth_idx * 2 + 1], reduce_result[0]);
	atomic_add(lambda_cloth_balloon[cloth_idx * 3 + 2], reduce_result[1]);
#else
	sa_cloth_volumn[cloth_idx * 2 + 1] = reduce_result[0]; // Sum of Volumn
	lambda_cloth_balloon[cloth_idx * 3 + 2] = reduce_result[1]; // Sum of Weighted Squared Norm Of Gradient
#endif
}

inline void solve_cloth_balloon_update_lambda(
	const uint cloth_idx, 
	PTR(float) sa_cloth_volumn, 
	PTR(float) lambda_cloth_balloon, 
	const float balloon_scale_rate, 
	const float stiffness_pressure, 
	const float substep_dt)
{
	const float alpha_tilde = 1.0f / (stiffness_pressure * substep_dt * substep_dt);
	const float rest_volumn = sa_cloth_volumn[cloth_idx * 2 + 0];
	const float curr_volumn = sa_cloth_volumn[cloth_idx * 2 + 1];

	const float lambda = lambda_cloth_balloon[cloth_idx * 3 + 0];
	const float curr_sum_wsng = lambda_cloth_balloon[cloth_idx * 3 + 2];
    const float C = curr_volumn - balloon_scale_rate * rest_volumn; 

	const float delta_lambda = (-C - alpha_tilde * lambda) / (curr_sum_wsng + alpha_tilde);
	lambda_cloth_balloon[cloth_idx * 3 + 0] = lambda + delta_lambda;
	lambda_cloth_balloon[cloth_idx * 3 + 1] = delta_lambda;
	lambda_cloth_balloon[cloth_idx * 3 + 2] = 0.0;
	sa_cloth_volumn[cloth_idx * 2 + 1] = 0.0;
}

inline void solve_cloth_balloon_modify_verts_pass(
	const uint vid, 
	PTR(uint) sa_vert_adj_faces,
	PTR(Float3) input_position, PTR(Float3) output_position, 
	const uint cloth_idx,
	PTR(float) lambda_cloth_balloon,
	PTR(float) sa_vert_mass_inv, PTR(Int3) sa_faces, const uint num_verts_total, const bool use_atomic
)
{
	const uint num_adj = sa_vert_adj_faces[vid];
	// const uint cloth_idx = sa_cloth_id[vid];
	const float delta_lambda = lambda_cloth_balloon[cloth_idx * 2 + 1];
	const float vert_weight = sa_vert_mass_inv[vid];
	Float3 dx = Zero3;

	for (uint j = 0; j < num_adj; j++)
	{
		const uint fid = sa_vert_adj_faces[(j + 1) * num_verts_total + vid];
		const Int3 face = sa_faces[fid];
		Int2 adj_vid;

		if (vid == face[0]) { adj_vid = makeInt2(face[1], face[2]); }
		if (vid == face[1]) { adj_vid = makeInt2(face[2], face[0]); }
		if (vid == face[2]) { adj_vid = makeInt2(face[0], face[1]); }

		const Float3 vert_pos[2] = {
			input_position[adj_vid[0]],
			input_position[adj_vid[1]],
		};
		const Float3 cross_value = cross_vec(vert_pos[0], vert_pos[1]);
		dx += cross_value;
	}
	
	dx = vert_weight * delta_lambda * dx;
	output_position[vid] += dx;

}




}


// Energy
namespace Constrains
{

namespace Energy
{

inline float compute_energy_inertia(
	const uint vid, const PTR(Float3) updatePosition, const PTR(SceneParams) scene, 
    const PTR(uchar) sa_is_fixed,
    const PTR(float) sa_vert_mass, const PTR(Float3) sa_start_position, const PTR(Float3) sa_vert_velocity)
{
    float implicit_dt = scene->implicit_dt;
	float num_stepstep = scene->num_substep;
	float h = implicit_dt / num_stepstep;
    Float3 x_new = updatePosition[vid];
    Float3 x_0 = sa_start_position[vid];
    Float3 v_0 = sa_vert_velocity[vid];
    float mass = sa_vert_mass[vid];
    Float3 gravity = scene->gravity;
    Float3 a_ext = gravity;
    const bool is_fixed = sa_is_fixed[vid];
    if (is_fixed) a_ext = Zero3;
    
    Float3 y = x_0 + h * v_0 + h * h * a_ext;
    return length_squared_vec(x_new - y) * mass / (2 * h * h);
}

inline float compute_energy_stretch_mass_spring(
	const uint eid, const PTR(Float3) input_position,
	PTR(Int2) sa_edges, PTR(float) sa_edge_rest_state_length, const float stiffness_spring) 
{
	Int2 edge = sa_edges[eid];
	float energy = 0.f;
	Float3 vert_pos[2] = {
		input_position[edge[0]],  
		input_position[edge[1]]  
	};
	const float rest_edge_length = sa_edge_rest_state_length[eid];

	Float3 diff = vert_pos[1] - vert_pos[0];
	float orig_lengthsqr = length_squared_vec(diff);
	if (orig_lengthsqr < 1e-8) { return 0.0f; }
	float l = sqrt_scalar(orig_lengthsqr);
	float l0 = rest_edge_length;
	float C = l - l0;

	energy = 0.5f * stiffness_spring * square_scalar(C);
	
	// stretch_mass_spring<true, false, false>(
	// 	&energy, nullptr, nullptr, 
	// 	vert_pos, rest_edge_length, stiffness_spring);
	return energy;
}



inline float compute_energy_bending(
	BendingType bending_type, const uint eid, const PTR(Float3) input_position, 
	 PTR(Int4) sa_bending_edges, PTR(Int2) sa_bending_edge_adj_faces, PTR(float)  sa_face_area, 
	PTR(Float4x4) sa_bending_edge_Q, PTR(float) sa_bending_edge_rest_state_angle,
	const float stiffness_bending_quadratic, const float stiffness_bending_DAB, const bool use_xpbd)
{
	Int4 edge = sa_bending_edges[eid];	
	const Float3 vert_pos[4] = {
		input_position[edge[0]],
		input_position[edge[1]],
		input_position[edge[2]],
		input_position[edge[3]]
	};
	
	float energy = 0.f;

	if (bending_type == Constrains::BendingTypeQuadratic) 
	{
		Float4x4 m_Q = sa_bending_edge_Q[eid];
		if (use_xpbd)
		{
			float C;
			core_bending_quadratic<true, false, false>(
				&C, nullptr, nullptr, 
				vert_pos, m_Q, 1.0f);
			energy = stiffness_bending_DAB * square_scalar(C);
		}
		else
		{
			core_bending_quadratic<true, false, false>(
				&energy, nullptr, nullptr, 
				vert_pos, m_Q, stiffness_bending_quadratic);
		}
	}   
	else if (bending_type == Constrains::BendingTypeDAB)
	{	
		if (use_xpbd)
		{
			Int4 edge = sa_bending_edges[eid];

			// const float eps = 1e-6;
			const float ke = stiffness_bending_DAB; // 弯曲刚度

			const uint i = edge[0];
			const uint j = edge[1];
			const uint k = edge[2];
			const uint l = edge[3];

			const Float3 x1 = input_position[i];
			const Float3 x2 = input_position[j];
			const Float3 x3 = input_position[k];
			const Float3 x4 = input_position[l];

			float C;
			{
				const FVector3f SharedEdgeNormalized = VectorGetSafeNormal(x2 - x1);

				const FVector3f P13CrossP23 = cross_vec(x1 - x3, x2 - x3);
				const float Normal1Len = length_vec(P13CrossP23);
				const FVector3f Normal1 = SafeDivide(P13CrossP23, Normal1Len);

				const FVector3f P24CrossP14 = cross_vec(x2 - x4, x1 - x4);
				const float Normal2Len = length_vec(P24CrossP14);
				const FVector3f Normal2 = SafeDivide(P24CrossP14, Normal2Len);

				const FVector3f N2CrossN1 = cross_vec(Normal2, Normal1);

				const float CosPhi = clamp_scalar(dot_vec(Normal1, Normal2), (float)(-0.999999), (float)(0.999999));
				const float SinPhi = clamp_scalar(dot_vec(N2CrossN1, SharedEdgeNormalized), (float)(-0.999999), (float)(0.999999));

				if (Normal1Len < FLOAT_SMALL_NUMBER || Normal2Len < FLOAT_SMALL_NUMBER || abs_scalar(CosPhi) < FLOAT_SMALL_NUMBER) return 0.0f;

				float Angle = atan2_scalar(SinPhi, CosPhi); // if CosPhi == 0, atan(sin/cos) -> nan, so we use safe atan2
				const float RestAngle = sa_bending_edge_rest_state_angle[eid];
				C = Angle - RestAngle;
			}
			energy = 0.5f * ke * square_scalar(C);
		}
		else 
		{
			energy = 0.0f;
		}
	}

	return energy;
}

inline float compute_energy_stress_neohookean(
	const uint tet_id, const PTR(Float3) input_position,
    const PTR(Int4) sa_tets, 
    const PTR(Float3x3) sa_Dm_inv, const PTR(float) sa_tet_volumn,
    const float m_first_lame, const float m_second_lame
    )
{
    const Int4 tet = sa_tets[tet_id];
    Float3 vert_pos[4] = {
        input_position[tet[0]],
        input_position[tet[1]],
        input_position[tet[2]],
        input_position[tet[3]]
    };
    
	const float tet_volumn = sa_tet_volumn[tet_id];
	const Float3x3 Dm_inv = sa_Dm_inv[tet_id];
	// const Float3x3 Dm_inv_T = transpose_mat(Dm_inv);
	// const THREAD Float3x3& dFdx = Dm_inv_T;

	const Float3x3 Ds = makeFloat3x3(
		vert_pos[1] - vert_pos[0], 
		vert_pos[2] - vert_pos[0], 
		vert_pos[3] - vert_pos[0]);
	const Float3x3 F = Ds * Dm_inv;

	const THREAD Float3& f1 = get(F, 0);
	const THREAD Float3& f2 = get(F, 1);
	const THREAD Float3& f3 = get(F, 2);

	const float tr = length_squared_vec(f1) + length_squared_vec(f2) + length_squared_vec(f3);
	const float J = determinant_mat(F); // det

	float local_energy = tet_volumn * (
		0.5f * m_first_lame * square_scalar(J - 1.0f) + 0.5f * m_second_lame * square_scalar(tr - 3.0f)
	); 
    return local_energy;
}

inline float compute_energy_collision_vv(
	const uint pair_idx, const PTR(Float3) input_position, 
	PTR(ProximityVV) collision_self_vv,
	PTR(uint) collision_count,
	const float thickness)
{
	if (pair_idx >= collision_count[0]) return 0.0f;

	const auto pair = collision_self_vv[pair_idx];
	const Int2 indices = pair.get_indices();
	const Float3 vert_pos[2] = {
		input_position[indices[0]],
		input_position[indices[1]],
	};

	const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	const Float3 diff = vert_pos[0] - vert_pos[1];
	const float proj = dot_vec(diff, normal);
	
	if (proj < thickness)
	{
		const float C_n = thickness - proj;
		return 0.5f * stiff * square_scalar(C_n);
	}
	else 
	{
		return 0.0f;
	}
	
}
inline float compute_energy_collision_vf(
	const uint pair_idx, 
	const PTR(Float3) input_position, 
	const PTR(Float3) obstacle_position, 
	PTR(ProximityVF) collision_self_vf,
	PTR(uint) collision_count,
	const float thickness)
{
	if (pair_idx >= collision_count[0]) return 0.0f;

	const auto pair = collision_self_vf[pair_idx];
	
	const Int4 indices = pair.get_indices();

	Float3 vert_pos = input_position[indices[0]];
	Float3 obs_vert_pos[3] = {
		obstacle_position[indices[1]],
		obstacle_position[indices[2]],
		obstacle_position[indices[3]]
	};
		
	const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	const Float3 bary = pair.get_face_weight();
	const Float3 xs = bary[0] * obs_vert_pos[0] + bary[1] * obs_vert_pos[1] + bary[2] * obs_vert_pos[2];
	const Float3 diff = vert_pos - xs;
	const float proj = dot_vec(diff, normal);

	if (proj < thickness)
	{
		const float C_n = thickness - proj;
		return 0.5f * stiff * square_scalar(C_n);
	}
	else 
	{
		return 0.0f;
	}

}

}


}


// Spatial Hashing
namespace SpatialHashing
{

CONSTEXPR float spacing = 0.01f;
CONSTEXPR float spacing_inv = 1.f / spacing;
CONSTEXPR uint table_devide_num_verts = 5;

inline int fn_intCoord(float coord) 
{ 
	return floor_scalar(coord * spacing_inv); 
	// return int(coord / spacing); 
};
inline uint fn_hashCoord(int x, int y, int z, const uint table_size) 
{  
	// fantasy function
	return  abs_scalar( int64(x *  92837111) ^ 
						int64(y * 689287499) ^ 
						int64(z * 283923481) ) % table_size; 
};

inline uint spatial_hashing_function(const THREAD Float3& pos, const THREAD uint& table_size)
{
    int h = fn_hashCoord(fn_intCoord(pos.x), fn_intCoord(pos.y), fn_intCoord(pos.z), table_size);
    return h;
}
inline uint get_morton_key_by_int_xyz(int int_x, int int_y, int int_z)
{
    // const uint key = (Morton32::expand_bits(int_x) << 2) | (Morton32::expand_bits(int_y) << 1) | Morton32::expand_bits(int_z);
    const uint key = (int_x << 20) | (int_y << 10) | (int_z);
    return key;
}
inline uint get_morton_key_by_position(CREF(Float3) vert_pos)
{
    int int_x = SpatialHashing::fn_intCoord(vert_pos.x); // 0~99
    int int_y = SpatialHashing::fn_intCoord(vert_pos.y);
    int int_z = SpatialHashing::fn_intCoord(vert_pos.z);
    return get_morton_key_by_int_xyz(int_x, int_y, int_z);
}
inline bool try_to_access_h_by_key(TREF(uint) h, PTR(uint) hash_table_belongs, CONST(uint) key, const uint table_size)
{
    if (hash_table_belongs[h] != key)
    {
        for (uint j = 1; j < 32; j++)
        {
            h = (h + 1) % table_size;
            if (hash_table_belongs[h] == key)
            {
                return true;
            }
        }
        return false;
    }
    else 
    {
        return true;
    }
}
inline Float3 get_normed_position_for_spatial_hashing(CREF(Float3) vert_pos, const AABB aabb)
{
	return vert_pos - aabb.min_pos;
}


inline void reset_collision_system(
	const uint vid,
	PTR(uint) collision_count,
	PTR(Int4) collision_count_indirect_cmd_buffer,

	PTR(uint) num_verts_in_cluster,
	PTR(Int4) uncolored_verts_indirect_cmd_buffer,
	PTR(uint) uncolored_verts_count,

	PTR(uint) hash_table_count,
	PTR(uint) hash_table_prefix,
	PTR(uint) hash_table_belongs,

	PTR(uint) vert_VV_num_broad_phase,
	PTR(uint) vert_VV_num_narrow_phase,

	const bool reset_coloring_system,
	const bool reset_hash_table,
	const bool reset_broad_narrow_count
)
{
	// if (vid == 0) 
	// {
	// 	hash_table_prefix[0] = 0;
	// }

	if (vid < 64) 
	{
		collision_count[vid] = 0;
		if (vid < 4) collision_count_indirect_cmd_buffer[vid] = make_indirect_command_buffer(0);
		if (reset_coloring_system)
		{
			num_verts_in_cluster[vid] = 0;
			uncolored_verts_indirect_cmd_buffer[vid] = make_indirect_command_buffer(0);
			uncolored_verts_count[vid] = (0);
		}
	}

	if (reset_hash_table) 
	{
		for (uint j = 0; j < table_devide_num_verts; j++) 
		{
			hash_table_count[table_devide_num_verts * vid + j] = 0;
			hash_table_prefix[table_devide_num_verts * vid + j] = 0;
			hash_table_belongs[table_devide_num_verts * vid + j] = -1u;
		}
	}
	if (reset_broad_narrow_count) // Not Run In Obstacle Collision (Actually NumThreads Does NOT Match)
	{
		vert_VV_num_broad_phase[vid] = 0;
		vert_VV_num_narrow_phase[vid] = 0;
	}
	
}
inline void reset_broad_narrow_count(
	const uint vid,
	PTR(uint) vert_VV_num_broad_phase,
	PTR(uint) vert_VV_num_narrow_phase
)
{
	{
		vert_VV_num_broad_phase[vid] = 0;
		vert_VV_num_narrow_phase[vid] = 0;
	}
}

inline void get_normalized_position(
	const uint vid, 
	PTR(Float3) curr_position, 
	PTR(Float3) norm_position, 
	const AABB global_aabb)
{
	const Float3 vert_pos = curr_position[vid];
	norm_position[vid] = (vert_pos - global_aabb.min_pos) ; // * dim_inv;
}
inline void set_hash_table_flag(
	const uint cell_id, 
	PTR(uchar) hash_table_flag, 
	PTR(uint) hash_table_cell_accessed_count)
{
	const uchar accessed_count = hash_table_cell_accessed_count[cell_id];
	if (accessed_count != 0)
	{
		hash_table_flag[cell_id] = accessed_count;
	}
	hash_table_cell_accessed_count[cell_id] = 0;
}

inline void fill_in_hash_table(
	const uint vid,
	PTR(Float3) sa_iter_position,
	PTR(uint) hash_table_count,
	PTR(uint) hash_table_belongs,
    PTR(uint) hash_table_cell_accessed_count,
	PTR(uint) hash_table_vert_offset,
	PTR(AABB) sa_block_aabb,
	const uint table_size)
{
	
	Float3 vert_pos = sa_iter_position[vid];
	vert_pos = get_normed_position_for_spatial_hashing(vert_pos, sa_block_aabb[0]);

	auto h = SpatialHashing::spatial_hashing_function(vert_pos, table_size);

	const uint key = get_morton_key_by_position(vert_pos);
	auto orig_value = atomic_cas(hash_table_belongs[h], -1u, key);

	atomic_add(hash_table_cell_accessed_count[h], 1u);

	if (orig_value != -1u && orig_value != key)
	{
		// const uint orig_h = h;
		for (uint j = 1; j < 32; j++) // Search For 32 Times
		{
			h = (h + 1) % table_size;
			orig_value = atomic_cas(hash_table_belongs[h], -1u, key);
			if (orig_value == -1u && orig_value != key)
			{

				// if (j > 5) 
				// {
				//     auto hh = SpatialHashing::spatial_hashing_function(vert_pos, table_size);
				//     fast_print("Access Same Cell With Offset", j);
				//     for (uint jj = 0; jj < j; jj++)
				//     {
				//         auto geted_key = hash_table_belongs[hh + jj];
				//         Int3 xyz = makeInt3((geted_key >> 16) & 0xFF, (geted_key >> 8) & 0xFF, (geted_key >> 0) & 0xFF);
				//         fast_format("    offset {} : {}    ({})", jj, SimString::Vec3_to_string(xyz), geted_key);
				//     }
				// }

				// hash_table_flag[h] = 0xFF;
				break;
			}
		}
	}

	uint offset = atomic_add(hash_table_count[h], 1);
	hash_table_vert_offset[vid] = offset;
}

inline uint scan_hash_table(
	const uint cell_id, 
	PTR(uint) hash_table_count)
{
	return hash_table_count[cell_id];
}

inline void insert_vert_into_hash_table(const uint vid,
	PTR(Float3) sa_iter_position,
	PTR(uint) hash_table_vert_offset,
	PTR(uint) hash_table_prefix,
	PTR(uint) hash_table_belongs,
	PTR(uint) hash_table,
	PTR(AABB) sa_block_aabb,
	const uint table_size)
{
	Float3 vert_pos = sa_iter_position[vid];
	vert_pos = get_normed_position_for_spatial_hashing(vert_pos, sa_block_aabb[0]);

	auto h = SpatialHashing::spatial_hashing_function(vert_pos, table_size);
	auto key = get_morton_key_by_position(vert_pos);
	
	try_to_access_h_by_key(h, hash_table_belongs, key, table_size);

	uint cell_prefix = hash_table_prefix[h];
	uint offset = hash_table_vert_offset[vid];

	uint insert_idx = cell_prefix + offset;
	hash_table[insert_idx] = vid;
}

// CONSTEXPR uint max_vv_num = 64;
// CONSTEXPR uint max_vf_num = 64;

// #define QUERY_RANGE_SCALE_TO_THICKNESS 1.0f
#define QUERY_RANGE_SCALE_TO_THICKNESS 1.2


inline void spatial_hashing_query_vv(const uint vid,
	PTR(Float3) sa_iter_position,
	PTR(Float3) sa_predict_position,
	PTR(uint) hash_table_count,
	PTR(uint) hash_table_prefix,
	PTR(uint) hash_table_belongs,
    PTR(uchar) hash_table_flag,
	PTR(uint) hash_table,
	PTR(uint) vert_VV_num_broad_phase,
	PTR(uint) broad_phase_list,
	PTR(AABB) sa_block_aabb,
	const uint table_size,
	const uint max_vv_num,
	const float query_range)
{
	// auto& list = broad_phase_list[vid];
	Float3 curr_pos = sa_iter_position[vid];
	curr_pos = get_normed_position_for_spatial_hashing(curr_pos, sa_block_aabb[0]);
	
	const float curr_vert_distance = query_range * QUERY_RANGE_SCALE_TO_THICKNESS;
	// const float curr_vert_distance = max_vert_distance[vid];

	int ix_min = fn_intCoord(curr_pos.x - curr_vert_distance); int ix_max = fn_intCoord(curr_pos.x + curr_vert_distance); 
	int iy_min = fn_intCoord(curr_pos.y - curr_vert_distance); int iy_max = fn_intCoord(curr_pos.y + curr_vert_distance); 
	int iz_min = fn_intCoord(curr_pos.z - curr_vert_distance); int iz_max = fn_intCoord(curr_pos.z + curr_vert_distance); 

	const uint start_idx = vid * max_vv_num;

#ifndef METAL_CODE
	std::unordered_set<uint> tmp; 
#endif

	uint curr_collision_count = 0;
	for (int x = ix_min; x <= ix_max; x++)
	{
		for (int y = iy_min; y <= iy_max; y++)
		{
			for (int z = iz_min; z <= iz_max; z++)
			{
				uint h = SpatialHashing::fn_hashCoord(x, y, z, table_size);
				
				const uint cell_accessed_count = hash_table_flag[h];
				if (cell_accessed_count != 0)
				{
					const auto key = get_morton_key_by_int_xyz(x, y, z);
					bool accessed = try_to_access_h_by_key(h, hash_table_belongs, key, table_size);
					if (accessed)
					{
#if false
						// // if (tmp.contains(h)) { break; } tmp.insert(h);
						if (tmp.contains(h)) 
						{ 
							fast_format_err("Vert {:4} Access Two Cell With The Same h ({}) (Pos = {}/{})", vid,h,  
								SimString::Vec3_to_string(simd::make_int3(x, y, z)), 
								SimString::Vec3_to_string(simd::make_int3(ix_min, iy_min, iz_min)),
								SimString::Vec3_to_string(simd::make_int3(ix_max, iy_max, iz_max))
								); 
							
							exit(0); 
						}
						tmp.insert(h);
#endif

						const uint cell_start = hash_table_prefix[h];
						const uint cell_size = hash_table_count[h];
						for (uint i = cell_start; i < cell_start + cell_size; i++)
						{
							const uint neighbor = hash_table[i];

							if (neighbor > vid) 
							{
								broad_phase_list[start_idx + curr_collision_count] = neighbor;
								curr_collision_count++;

								if (curr_collision_count == max_vv_num - 1)  { vert_VV_num_broad_phase[vid] = max_vv_num - 1; return; }
							}
						}
					}

					
				}
			}
		}
	}

	vert_VV_num_broad_phase[vid] = curr_collision_count;
}



}


// Prepare Position For Collision Detection
namespace Constrains
{

inline void prepare_position_for_collision_detection(
	const uint vid,
	PTR(Float3) sa_position_cloth,
	PTR(Float3) sa_position_tet,
	PTR(uint) sa_surface_verts,
	PTR(Float3) sa_position_for_detection_bg,
	PTR(Float3) sa_position_for_detection_ed,
	const uint num_verts_cloth
)
{	
	const Float3 vert_pos = vid < num_verts_cloth ? sa_position_cloth[vid] :
													sa_position_tet[sa_surface_verts[vid - num_verts_cloth]];
	sa_position_for_detection_bg[vid] = vert_pos;
	sa_position_for_detection_ed[vid] = vert_pos;
}



}

namespace NarrowPhase
{

///////////////////////////////////////////////////////////////////////
// find the distance from a line segment (v1, v2) to a point (v0)
///////////////////////////////////////////////////////////////////////
inline REAL pointLineDistance(CREF(Float3) v0, CREF(Float3) v1, CREF(Float3) v2, 
	CREF(Float3) e1hat, CONST(float) projection)
{
	const Float3 e0 = v0 - v1;
	const Float3 e1 = v2 - v1;
	// const Float3 e1hat = normalize_vec(e1);
	// const REAL projection = dot_vec(e0, e1hat);

	// if it projects onto the line segment, use that length
	if (projection > 0.0 && projection < length_vec(e1))
	{
		const Float3 normal = e0 - projection * e1hat;
		return length_vec(normal);
	}

	// if it doesn't, find the point-point distances
	const REAL diff01 = length_vec(v0 - v1);
	const REAL diff02 = length_vec(v0 - v2);

	return (diff01 < diff02) ? diff01 : diff02;
}

///////////////////////////////////////////////////////////////////////
// get the linear interpolation coordinates from v0 to the line segment
// between v1 and v2
///////////////////////////////////////////////////////////////////////
inline Float2 getLerp(CREF(Float3) v0, CREF(Float3) v1, CREF(Float3) v2,
	CREF(Float3) e1hat, CONST(float) projection)
{
	// const Float3 e0 = v0 - v1;
	const Float3 e1 = v2 - v1;
	// const Float3 e1hat = normalize_vec(e1);
	// const REAL projection = dot_vec(e0, e1hat);

	if (projection < 0.0)
		return makeFloat2(1.0f, 0.0f);

	if (projection >= length_vec(e1))
		return makeFloat2(0.0f, 1.0f);

	const REAL ratio = projection / length_vec(e1);
		return makeFloat2(1.0f - ratio, ratio);
}

template <bool ignore_vert_projection_not_in_triangle>
inline bool template_vf(
	const Float3 vert_pos[4], CONST(float) min_distance,
	TREF(float) distance, TREF(Float3) t, TREF(Float4) weight)
{
    const Float3 e1 = vert_pos[2] - vert_pos[1];
    const Float3 e2 = vert_pos[3] - vert_pos[2];
    const Float3 e3 = vert_pos[1] - vert_pos[3];
    const Float3 n = cross_vec(e1, e2);

    // Bary Coordinates
    const Float3 na = cross_vec(vert_pos[3] - vert_pos[2], vert_pos[0] - vert_pos[2]);
    const Float3 nb = cross_vec(vert_pos[1] - vert_pos[3], vert_pos[0] - vert_pos[3]);
    const Float3 nc = cross_vec(vert_pos[2] - vert_pos[1], vert_pos[0] - vert_pos[1]);
    
    Float3 barycentric = makeFloat3(
        dot_vec(n, na) / dot_vec(n, n),
        dot_vec(n, nb) / dot_vec(n, n),
        dot_vec(n, nc) / dot_vec(n, n)
    );

	const float barySum = abs_scalar(barycentric[0]) + abs_scalar(barycentric[1]) + abs_scalar(barycentric[2]);
	const bool vert_projection_not_in_triangle = barySum - 1.0 > 1e-8; // barySum > 1
	if (vert_projection_not_in_triangle)
    {
		// Find The Closest Distance Betweeen Vert And Face's Edges
		if (!ignore_vert_projection_not_in_triangle) 
		{
			const Float3 e1hat_12 = normalize_vec(e1);
			const Float3 e1hat_23 = normalize_vec(e2);
			const Float3 e1hat_31 = normalize_vec(e3);
			const float projection_01 = dot_vec(vert_pos[0] - vert_pos[1], e1hat_12);
			const float projection_02 = dot_vec(vert_pos[0] - vert_pos[2], e1hat_23);
			const float projection_03 = dot_vec(vert_pos[0] - vert_pos[3], e1hat_31);

			// pointProjects NOT InsideTriangle
			REAL distance12 = pointLineDistance(vert_pos[0], vert_pos[1], vert_pos[2], e1hat_12, projection_01);
			REAL distance23 = pointLineDistance(vert_pos[0], vert_pos[2], vert_pos[3], e1hat_23, projection_02);
			REAL distance31 = pointLineDistance(vert_pos[0], vert_pos[3], vert_pos[1], e1hat_31, projection_03);

			if (distance12 <= distance23 && distance12 <= distance31)
			{
				Float2 lerp = getLerp(vert_pos[0], vert_pos[1], vert_pos[2], e1hat_12, projection_01);
				barycentric = makeFloat3(lerp[0], lerp[1], 0.0f);
			}
			else if (distance23 <= distance12 && distance23 <= distance31)
			{
				Float2 lerp = getLerp(vert_pos[0], vert_pos[2], vert_pos[3], e1hat_23, projection_02);
				barycentric = makeFloat3(0.0f, lerp[0], lerp[1]);
			}
			else
			{
				Float2 lerp = getLerp(vert_pos[0], vert_pos[3], vert_pos[1], e1hat_31, projection_03);
				barycentric = makeFloat3(lerp[1], 0.0f, lerp[0]);
			}
		}
		else 
		{
			return false;
		}
    }

	// if (distance < SimContactEnergy::_eps && distance > SimContactEnergy::_tooSmall) 
	
	const Float3 xs = barycentric[0] * vert_pos[1] + barycentric[1] * vert_pos[2] + barycentric[2] * vert_pos[3];
    t = vert_pos[0] - xs;
    distance = length_vec(t);
	if (distance < min_distance && distance > 1e-8)
	{
		// weight = makeFloat4(1.f, -barycentric[0], -barycentric[1], -barycentric[2]);
		weight = makeFloat4(1.f, barycentric[0], barycentric[1], barycentric[2]);
		return true;
	}
	else 
	{
		return false;
	}

}


inline void narrow_phase_vv_self_collision_from_collision_pair(const uint index,
	PTR(Float3) sa_detection_position_bg,
	PTR(Float3) sa_detection_position_ed,
	PTR(Float3) sa_detection_rest_position, 
	// PTR(uint) sa_indices_map, 

	PTR(uint) vert_VV_num_broad_phase,
	PTR(uint) broad_phase_list,
	
	PTR(Int2) narrow_phase_list_indices_vv,
	PTR(ProximityVV) narrow_phase_list_pair_vv,
	PTR(uint) collision_count,
	PTR(Int4) self_collision_indirect_cmd_buffer,
	PTR(uint) vert_VV_num_narrow_phase,
	PTR(uint) vert_VV_prefix_narrow_phase,
	PTR(uchar) collision_pair_offset_in_vert,
	PTR(uint) vert_adj_pairs,

	const float thickness_1,
	const float thickness_2,
	const float stiffness_collision) 
{
	if (index >= self_collision_indirect_cmd_buffer[1][3]) return;

	const uint left = broad_phase_list[index * 2 + 0];
	const uint right = broad_phase_list[index * 2 + 1];

	if (left < right) // VV May Be Redundant 
	{
		const float max_distance = (thickness_1 + thickness_2) * QUERY_RANGE_SCALE_TO_THICKNESS;
		const float max_squared_distance = max_distance * max_distance;
		const float max_rest_squared_distance = max_squared_distance ; // * 4.0;

		const Float3 a_curr_pos = sa_detection_position_bg[left];
		const Float3 a_rest_pos = sa_detection_rest_position[left];
		 
		{
			const Float3 b_curr_pos = sa_detection_position_bg[right];
			const Float3 diff = a_curr_pos - b_curr_pos;

			const float curr_squared_length = length_squared_vec(diff); // length_squared_vec(a_curr_pos - b_curr_pos);
			if (curr_squared_length < max_squared_distance && curr_squared_length > Epsilon)
			{
				const Float3 b_rest_pos = sa_detection_rest_position[right];
				const float rest_squared_length = length_squared_vec(a_rest_pos - b_rest_pos);

				if (rest_squared_length > max_rest_squared_distance) 
				{
					uint idx = atomic_add(collision_count[0], 1);
					narrow_phase_list_indices_vv[idx] = makeInt2(left, right);

					const float average_area = 1.0f;

					Float3 normal = diff / sqrt_scalar(curr_squared_length);;
					narrow_phase_list_pair_vv[idx] = ProximityVV(left, right, stiffness_collision, average_area, normal);

					uint offset_1 = atomic_add(vert_VV_num_narrow_phase[left], 1);
					uint offset_2 = atomic_add(vert_VV_num_narrow_phase[right], 1);
					collision_pair_offset_in_vert[2 * idx + 0] = offset_1;
					collision_pair_offset_in_vert[2 * idx + 1] = offset_2;
				}
			} 
		}
	}
}
inline void narrow_phase_vv_self_collision_codim_from_collision_pair(const uint index,
	PTR(Float3) sa_detection_position_left,
	PTR(Float3) sa_detection_position_right,

	PTR(uint) vert_VV_num_broad_phase,
	PTR(uint) broad_phase_list,
	
	PTR(Int2) narrow_phase_list_indices_vv,
	PTR(ProximityVV) narrow_phase_list_pair_vv,
	PTR(uint) collision_count,
	PTR(Int4) self_collision_indirect_cmd_buffer,
	PTR(uint) vert_VV_num_narrow_phase,
	PTR(uint) vert_VV_prefix_narrow_phase,
	PTR(uchar) collision_pair_offset_in_vert,
	PTR(uint) vert_adj_pairs,

	const uint object2_vid_prefix,
	const float thickness_1,
	const float thickness_2,
	const float stiffness_collision) 
{
	if (index >= self_collision_indirect_cmd_buffer[1][3]) return;

	const uint left = broad_phase_list[index * 2 + 0];
	const uint right = broad_phase_list[index * 2 + 1];

	// if (left < right) // VV May Be Redundant 
	{
		const float max_distance = (thickness_1 + thickness_2) * QUERY_RANGE_SCALE_TO_THICKNESS;
		const float max_squared_distance = max_distance * max_distance;

		const Float3 a_curr_pos = sa_detection_position_left[left];
		
		{
			const Float3 b_curr_pos = sa_detection_position_right[right];
			const Float3 diff = a_curr_pos - b_curr_pos;

			const float curr_squared_length = length_squared_vec(diff); // length_squared_vec(a_curr_pos - b_curr_pos);
			if (curr_squared_length < max_squared_distance && curr_squared_length > Epsilon)
			{
				{
					const uint right_global = right + object2_vid_prefix;

					uint idx = atomic_add(collision_count[0], 1);
					narrow_phase_list_indices_vv[idx] = makeInt2(left, right_global);

					const float average_area = 1.0f;
					Float3 normal = diff / sqrt_scalar(curr_squared_length);;
					narrow_phase_list_pair_vv[idx] = ProximityVV(left, right_global, stiffness_collision, average_area, normal);

					uint offset_1 = atomic_add(vert_VV_num_narrow_phase[left], 1);
					uint offset_2 = atomic_add(vert_VV_num_narrow_phase[right_global], 1);
					collision_pair_offset_in_vert[2 * idx + 0] = offset_1;
					collision_pair_offset_in_vert[2 * idx + 1] = offset_2;
				}
			} 
		}
	}
}

inline void narrow_phase_vf_obstacle_collision_from_collision_pair(
	const uint index, 
	
	PTR(Float3) sa_detection_position_bg,
	PTR(Float3) sa_detection_position_ed,
	PTR(Float3) sa_obstacle_position,
	
	PTR(float) sa_detection_vert_area,
	PTR(Float3) sa_obs_vert_normal,
	PTR(Float3) sa_obs_face_normal,
	PTR(Int3) sa_obstacle_faces,

	PTR(uint) vert_VV_num_broad_phase,
	PTR(uint) broad_phase_list,

	PTR(Int4) narrow_phase_list_indices_vf,
	PTR(ProximityVF) narrow_phase_list_pair_vf,
	PTR(uint) collision_count,
	PTR(Int4) indirect_command_buffer,
	
	PTR(uint) vert_VV_num_narrow_phase,
	PTR(uint) vert_VV_prefix_narrow_phase,
	PTR(uchar) collision_pair_offset_in_vert,
	PTR(uint) vert_adj_verts_vv,

	const uint max_vf_broad_phase_num,
	const uint max_vf_narrow_phase_num,
	const float thickness_1,
	const float thickness_2,
	const float stiffness_collision)
{
	const uint total_broad_phase = indirect_command_buffer[1][3];
	if (index >= total_broad_phase) return;

	const uint vid = broad_phase_list[index * 2 + 0];
	const uint adj_fid = broad_phase_list[index * 2 + 1];
	const float min_distance = (thickness_1 + thickness_2) * QUERY_RANGE_SCALE_TO_THICKNESS;

    const Float3 a_curr_pos = sa_detection_position_bg[vid];
	// const float vert_area = sa_detection_vert_area[vid];
	const float vert_area = 1.0f;

	const Int3 adj_face = sa_obstacle_faces[adj_fid];
	Float3 _vertices[4] = { 
		a_curr_pos, 
		sa_obstacle_position[adj_face[0]], 
		sa_obstacle_position[adj_face[1]], 
		sa_obstacle_position[adj_face[2]] 
	};

	const bool ignore_vert_projection_not_in_triangle = false;
	float distance; Float3 t; Float4 weight;
	const bool is_collision = template_vf<ignore_vert_projection_not_in_triangle>(_vertices, min_distance, distance, t, weight);
	if (is_collision)
	{
		const Int4 tet = makeInt4(vid, adj_face[0], adj_face[1], adj_face[2]);
		const uint offset = atomic_add(vert_VV_num_narrow_phase[vid], 1);
		// if (offset >= max_vf_narrow_phase_num)
		// {
		// 	atomic_sub(vert_VV_num_narrow_phase[vid], 1);
		// } 
		// else 
		{
			const uint pair_idx = atomic_add(collision_count[0], 1);
		
			narrow_phase_list_indices_vf[pair_idx] = tet;   
			const Float3 normal = sa_obs_face_normal[adj_fid];
			narrow_phase_list_pair_vf[pair_idx] = ProximityVF(vid, adj_face, weight, stiffness_collision, vert_area, normal);
			collision_pair_offset_in_vert[pair_idx] = offset;
		}
	}
	
	// {
	// 	const Int3 adj_face = sa_obstacle_faces[adj_fid];
	// 	Float3 _vertices[4] = { 
	// 		a_curr_pos, 
	// 		sa_obstacle_position[adj_face[0]], 
	// 		sa_obstacle_position[adj_face[1]], 
	// 		sa_obstacle_position[adj_face[2]] 
	// 	};

	// 	const bool ignore_vert_projection_not_in_triangle = false;
	// 	float distance; Float3 t; Float4 weight;
	// 	const bool is_collision = template_vf<ignore_vert_projection_not_in_triangle>(_vertices, min_distance, distance, t, weight);

	// 	if (is_collision)
	// 	{
	// 		const Int4 tet = makeInt4(vid, adj_face[0], adj_face[1], adj_face[2]);
	// 		const uint offset = atomic_add(vert_VV_num_narrow_phase[vid], 1); 

	// 		const uint pair_idx = atomic_add(collision_count[0], 1);

	// 		narrow_phase_list_indices_vf[pair_idx] = tet;   
	// 		const Float3 normal = sa_obs_face_normal[adj_fid];
	// 		narrow_phase_list_pair_vf[pair_idx] = ProximityVF(vid, adj_face, weight, stiffness_collision, vert_area, normal);
	// 		collision_pair_offset_in_vert[pair_idx] = offset;
	// 	}
	// }
}


inline uint narrow_phase_scan_get_num(const uint vid,
	PTR(uint) vert_VV_num_narrow_phase)
{
	const uint num_vf = vert_VV_num_narrow_phase[vid];
	// if (num_vf > 96) { vert_VV_num_narrow_phase[vid] = 0; return 0; } // Filtering The Max Collision Num
	return num_vf; 
}
inline void narrow_phase_scan_write_prefix(const uint vid, 
	CONST(uint) scan_result, CONST(uint) self_result, 
	PTR(uint) vert_VV_prefix_narrow_phase)
{
    vert_VV_prefix_narrow_phase[vid + 1] = scan_result; 
}
inline uint fn_atomic_add_num_collision_in_block(
	PTR(uint) collision_count, const uint num_vf_in_block)
{
    return atomic_add(collision_count[6], num_vf_in_block);
}

template <typename ElementType, uint N = Meta::get_vec_length<ElementType>()>
inline void self_collision_fill_in(const uint pair_idx,
	PTR(uchar) collision_pair_offset_in_vert, 
	PTR(ElementType) narrow_phase_list,
	PTR(uint) vert_prefix_narrow_phase,
	PTR(uint) vert_adj_elements)
{
	const ElementType tet = narrow_phase_list[pair_idx];

	// const Int4 insert_idx = prefix + offset;
	for (uint j = 0; j < N; j++) 
	{
		const uint offset = collision_pair_offset_in_vert[N * pair_idx + j];
		const uint prefix = vert_prefix_narrow_phase[tet[j]];
		const uint insert_idx = prefix + offset;
		vert_adj_elements[insert_idx] = pair_idx;
	}
}
template <typename ElementType> //, uint N = Meta::get_vec_length<ElementType>()>
inline void obstacle_collision_fill_in(const uint pair_idx,
	PTR(uchar) collision_pair_offset_in_vert, 
	PTR(ElementType) narrow_phase_list,
	PTR(uint) vert_prefix_narrow_phase,
	PTR(uint) vert_adj_elements)
{
	const ElementType tet = narrow_phase_list[pair_idx];
	{
		const uint offset = collision_pair_offset_in_vert[pair_idx];
		const uint prefix = vert_prefix_narrow_phase[tet[0]];
		const uint insert_idx = prefix + offset;
		vert_adj_elements[insert_idx] = pair_idx;
	}
	// for (uint j = 0; j < 1; j++) 
	// {
	// 	const uint offset = collision_pair_offset_in_vert[N * pair_idx + j];
	// 	const uint prefix = vert_prefix_narrow_phase[tet[j]];
	// 	const uint insert_idx = prefix + offset;
	// 	vert_adj_elements[insert_idx] = pair_idx;
	// }
}



}

// Collision Constraints
namespace Constrains
{

// IPC / VBD Friction
// {
// 	const Float3 partial_xc = (vert_pos[1] - start_vert_pos[1]) - (vert_pos[0] - start_vert_pos[0]);
// 	const Float3 t_hat = normalize_vec(diff);
// 	const Float3 b_hat = normalize_vec(cross_vec(t_hat, normal));
// 	const Float2x3 Tc = makeFloat2x3(t_hat, b_hat);
// 	const Float2 u_c = transpose_mat(Tc) * partial_xc; const float len_uc = length_vec(u_c); const Float2 dir_uc = u_c / len_uc;
// 	const Float3 lambda_ci = stiffness_collision * cross_vec(cross_vec(diff, normal), normal);
// 	const auto f_ci = -stiffness_friction * lambda_ci * Tc * f1(len_uc, substep_dt, 1.0f) * dir_uc;
	
// 	float lambda_f = lambda_self_collision_friction[pair_idx];
// 	float delta_lambda_f = Constrains::Core::get_additional_delta_xpbd_2(
// 		dx[0], dx[1], gradient, -gradient, vert_inv_mass[0], vert_inv_mass[1], 
// 		energy, lambda_n, alpha_tilde);

// 	//  To satisfy Coulomb's law that the frictional force should be limited by the normal force 
// 	//  we clamp the frictional Lagrange multiplier updates as follows
// 	delta_lambda_f = min_scalar(stiffness_friction * delta_lambda_f, delta_lambda_n);
// }



// 
// Self Collision : Per Constraint or Per Vert
//

inline float f1(const float u, const float h, const float epsilon_v)
{
	if (u > 0.0f && u < epsilon_v)
	{
		float tmp = u / (h * epsilon_v);
		return 2.0f * (tmp) - tmp * tmp;
	}
	else 
	{
		return 1.0f;
	}
}

// inline void get_friction(CREF(Float3) normal, CREF(Float3) relative_vel)
// {
// }

// Self-Collision
inline void solve_self_collision_vv_per_collision_pair_template_cloth(
	const uint pair_idx, 
	PTR(Float3) substep_start_position_cloth, 
	PTR(Float3) substep_start_position_tet, 
	PTR(Float3) iter_position_cloth, 
	PTR(Float3) iter_position_tet, 
	PTR(Float3) output_position_cloth, 
	PTR(Float3) output_position_tet,

	PTR(uint) sa_surface_verts, 
	PTR(float) sa_vert_mass_inv_cloth, 
	PTR(float) sa_vert_mass_inv_tet, 
	PTR(ATOMIC_FLAG) sa_vert_mutex_cloth,
	PTR(ATOMIC_FLAG) sa_vert_mutex_tet,

	PTR(ProximityVV) self_collision_pair_vv,
	PTR(float) lambda_self_collision,
	PTR(float) lambda_self_collision_friction,

	const float substep_dt, 
	const bool use_atomic,
	const float thickness,
	const float stiffness_collision,
	const float stiffness_friction,
	const uint num_verts_cloth)
{
	const auto pair = self_collision_pair_vv[pair_idx];
	Int2 indices = pair.get_indices();
	const Float3 vert_pos[2] = {
		iter_position_cloth[indices[0]],
		iter_position_cloth[indices[1]],
	};
	const Float3 start_vert_pos[2] = {
		substep_start_position_cloth[indices[0]],
		substep_start_position_cloth[indices[1]],
	};
	const Float3 vert_vel[2] = {
		(vert_pos[0] - start_vert_pos[0]) / substep_dt,
		(vert_pos[1] - start_vert_pos[1]) / substep_dt,
	};
	const float vert_inv_mass[2] = {
		sa_vert_mass_inv_cloth[indices[0]],
		sa_vert_mass_inv_cloth[indices[1]],
	};
	// const Float2 weight = pair.get_weight();
	const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	const Float3 diff = vert_pos[0] - vert_pos[1];
	const float proj = dot_vec(diff, normal);

	if (proj < thickness)
	{
		Float3 dx[2] = {Zero3, Zero3};
		const float C_n = thickness - proj;

		// Elastic Collision
		// CONSTIF (false)
		{
			const float alpha_tilde_n = 1.0f / (stiff * substep_dt * substep_dt);
			// const float alpha_tilde_f = 1.0f / (stiffness_friction * substep_dt * substep_dt);

			float energy_n; Float3 gradient_n;
			get_collision_C_and_dCdx(C_n, normal, stiff, energy_n, gradient_n);
			const float lambda_n = lambda_self_collision[pair_idx];

			const float delta_lambda_n = Constrains::Core::get_delta_xpbd_2(
				dx[0], dx[1], gradient_n, -gradient_n, vert_inv_mass[0], vert_inv_mass[1], 
				energy_n, lambda_n, alpha_tilde_n);
			
			lambda_self_collision[pair_idx] += delta_lambda_n;

			// Friction
			{			
				Float3 v1 = vert_vel[0];
				Float3 v2 = vert_vel[1];
				Float3 v_avg = 0.5f * (v1 + v2);
				const float friction_damping = stiffness_friction;
				dx[0] += friction_damping * substep_dt * (v_avg - v1);
				dx[1] += friction_damping * substep_dt * (v_avg - v2);
			}
		}
		
		// Friction May Distroy The Fix-Point Info, So We Need To use InvMass Again...

		if (vert_inv_mass[0] != 0.0f) { iter_position_cloth[indices[0]] += dx[0]; } 
		if (vert_inv_mass[1] != 0.0f) { iter_position_cloth[indices[1]] += dx[1]; } 
	}
}
inline void solve_self_collision_vv_per_collision_pair_template_tet(
	const uint pair_idx, 
	PTR(Float3) substep_start_position_cloth, 
	PTR(Float3) substep_start_position_tet, 
	PTR(Float3) iter_position_cloth, 
	PTR(Float3) iter_position_tet, 
	PTR(Float3) output_position_cloth, 
	PTR(Float3) output_position_tet,

	PTR(uint) sa_surface_verts, 
	PTR(float) sa_vert_mass_inv_cloth, 
	PTR(float) sa_vert_mass_inv_tet, 
	PTR(ATOMIC_FLAG) sa_vert_mutex_cloth,
	PTR(ATOMIC_FLAG) sa_vert_mutex_tet,

	PTR(ProximityVV) self_collision_pair_vv,
	PTR(float) lambda_self_collision,
	PTR(float) lambda_self_collision_friction,

	const float substep_dt, 
	const bool use_atomic,
	const float thickness,
	const float stiffness_collision,
	const float stiffness_friction,
	const uint num_verts_cloth)
{
	const auto pair = self_collision_pair_vv[pair_idx];
	Int2 indices = pair.get_indices();
	indices[0] = sa_surface_verts[indices[0]];
	indices[1] = sa_surface_verts[indices[1]];
	const Float3 vert_pos[2] = {
		iter_position_tet[indices[0]],
		iter_position_tet[indices[1]],
	};
	const Float3 start_vert_pos[2] = {
		substep_start_position_tet[indices[0]],
		substep_start_position_tet[indices[1]],
	};
	const Float3 vert_vel[2] = {
		(vert_pos[0] - start_vert_pos[0]) / substep_dt,
		(vert_pos[1] - start_vert_pos[1]) / substep_dt,
	};
	const float vert_inv_mass[2] = {
		sa_vert_mass_inv_tet[indices[0]],
		sa_vert_mass_inv_tet[indices[1]],
	};
	// const Float2 weight = pair.get_weight();
	const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	const Float3 diff = vert_pos[0] - vert_pos[1];
	const float proj = dot_vec(diff, normal);

	if (proj < thickness)
	{
		Float3 dx[2] = {Zero3, Zero3};
		const float C_n = thickness - proj;

		// Elastic Collision
		// CONSTIF (false)
		{
			const float alpha_tilde_n = 1.0f / (stiff * substep_dt * substep_dt);
			// const float alpha_tilde_f = 1.0f / (stiffness_friction * substep_dt * substep_dt);

			float energy_n; Float3 gradient_n;
			get_collision_C_and_dCdx(C_n, normal, stiff, energy_n, gradient_n);
			const float lambda_n = lambda_self_collision[pair_idx];

			const float delta_lambda_n = Constrains::Core::get_delta_xpbd_2(
				dx[0], dx[1], gradient_n, -gradient_n, vert_inv_mass[0], vert_inv_mass[1], 
				energy_n, lambda_n, alpha_tilde_n);
			
			lambda_self_collision[pair_idx] += delta_lambda_n;

			// Friction
			{			
				Float3 v1 = vert_vel[0];
				Float3 v2 = vert_vel[1];
				Float3 v_avg = 0.5f * (v1 + v2);
				const float friction_damping = stiffness_friction;
				dx[0] += friction_damping * substep_dt * (v_avg - v1);
				dx[1] += friction_damping * substep_dt * (v_avg - v2);
			}
		}
		
		// Friction May Distroy The Fix-Point Info, So We Need To use InvMass Again...

		if (vert_inv_mass[0] != 0.0f) { iter_position_tet[indices[0]] += dx[0]; }
		if (vert_inv_mass[1] != 0.0f) { iter_position_tet[indices[1]] += dx[1]; }
		
	}
}
inline void solve_self_collision_vv_per_collision_pair_template_cross(
	const uint pair_idx, 
	PTR(Float3) substep_start_position_cloth, 
	PTR(Float3) substep_start_position_tet, 
	PTR(Float3) iter_position_cloth, 
	PTR(Float3) iter_position_tet, 
	PTR(Float3) output_position_cloth, 
	PTR(Float3) output_position_tet,

	PTR(uint) sa_surface_verts, 
	PTR(float) sa_vert_mass_inv_cloth, 
	PTR(float) sa_vert_mass_inv_tet, 
	PTR(ATOMIC_FLAG) sa_vert_mutex_cloth,
	PTR(ATOMIC_FLAG) sa_vert_mutex_tet,

	PTR(ProximityVV) self_collision_pair_vv,
	PTR(float) lambda_self_collision,
	PTR(float) lambda_self_collision_friction,

	const float substep_dt, 
	const bool use_atomic,
	const float thickness,
	const float stiffness_collision,
	const float stiffness_friction,
	const uint object2_vid_prefix)
{
	const auto pair = self_collision_pair_vv[pair_idx];
	Int2 indices = pair.get_indices();
	indices[1] = sa_surface_verts[indices[1] - object2_vid_prefix];
	const Float3 vert_pos[2] = {
		iter_position_cloth[indices[0]],
		iter_position_tet[indices[1]],
	};
	const Float3 start_vert_pos[2] = {
		substep_start_position_cloth[indices[0]],
		 substep_start_position_tet[indices[1]],
	};
	const Float3 vert_vel[2] = {
		(vert_pos[0] - start_vert_pos[0]) / substep_dt,
		(vert_pos[1] - start_vert_pos[1]) / substep_dt,
	};
	const float vert_inv_mass[2] = {
		sa_vert_mass_inv_cloth[indices[0]],
		sa_vert_mass_inv_tet[indices[1]],
	};
	// const Float2 weight = pair.get_weight();
	const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	const Float3 diff = vert_pos[0] - vert_pos[1];
	const float proj = dot_vec(diff, normal);

	if (proj < thickness)
	{
		Float3 dx[2] = {Zero3, Zero3};
		const float C_n = thickness - proj;

		// Elastic Collision
		// CONSTIF (false)
		{
			const float alpha_tilde_n = 1.0f / (stiff * substep_dt * substep_dt);
			const float alpha_tilde_f = 1.0f / (stiffness_friction * substep_dt * substep_dt);

			float energy_n; Float3 gradient_n;
			get_collision_C_and_dCdx(C_n, normal, stiff, energy_n, gradient_n);
			const float lambda_n = lambda_self_collision[pair_idx];

			// const float damping_rate = 5.0; 
			// const float gamma = damping_rate / (stiff * substep_dt);

			// const float denom_n = 
			// 	(1.0 + gamma) * (vert_inv_mass[0] * length_squared_vec(gradient_n) + vert_inv_mass[1] * length_squared_vec(gradient_n)) + 
			// 	alpha_tilde_n;

			// const float numer_n = -(energy_n + alpha_tilde_n * lambda_n + 
			// 	gamma * substep_dt * ( dot_vec(gradient_n, vert_vel[0]) + dot_vec(-gradient_n, vert_vel[1]) ));

			// const float delta_lambda_n = numer_n / denom_n;
			// if (denom_n != 0.f)
			// {
			// 	dx[0] =  delta_lambda_n * vert_inv_mass[0] * gradient_n;
			// 	dx[1] = -delta_lambda_n * vert_inv_mass[1] * gradient_n;
			// }

			const float delta_lambda_n = Constrains::Core::get_delta_xpbd_2(
				dx[0], dx[1], gradient_n, -gradient_n, vert_inv_mass[0], vert_inv_mass[1], 
				energy_n, lambda_n, alpha_tilde_n);
			
			lambda_self_collision[pair_idx] += delta_lambda_n;

			// Friction
			{			
				Float3 v1 = vert_vel[0];
				Float3 v2 = vert_vel[1];
				Float3 v_avg = 0.5f * (v1 + v2);
				const float friction_damping = stiffness_friction;
				dx[0] += friction_damping * substep_dt * (v_avg - v1);
				dx[1] += friction_damping * substep_dt * (v_avg - v2);
			}
			CONSTIF (false)
			{
				const Float3 relative_vel_in_tangent_space = (vert_vel[0] - project_vec(vert_vel[0], normal)) - (vert_vel[1] - project_vec(vert_vel[1], normal));
				// const Float3 relative_vel = vert_vel[0] - vert_vel[1];
				// const Float3 relative_vel_in_tangent_space = relative_vel - dot_vec(relative_vel, normal) * normal; // == project_vec(relative_vel, normal); ;

				const Float3 delta_x_in_tangent_space = substep_dt * relative_vel_in_tangent_space;
				const THREAD Float3& x_t = delta_x_in_tangent_space;

				const float C_f = length_vec(x_t);
				if (C_f > Epsilon)
				{
					const Float3 dir_f = x_t / C_f;
				
					float energy_f; Float3 gradient_f;
					get_collision_C_and_dCdx(C_f, dir_f, stiffness_friction, energy_f, gradient_f);

					const float lambda_f = lambda_self_collision_friction[pair_idx];
					const float denom_f = 
						vert_inv_mass[0] * length_squared_vec(gradient_f) +
						vert_inv_mass[1] * length_squared_vec(gradient_f) + alpha_tilde_f;

					if (denom_f != 0)
					{
						float delta_lambda_f = (-C_f - alpha_tilde_f * lambda_f) / (denom_f); 
						delta_lambda_f = min_scalar(delta_lambda_f, stiffness_friction * delta_lambda_n);
						
						dx[0] += -delta_lambda_f * vert_inv_mass[0] * gradient_f;
						dx[1] +=  delta_lambda_f * vert_inv_mass[1] * gradient_f;

						lambda_self_collision_friction[pair_idx] += delta_lambda_f;
					}
				}
			}
		}
		
		// Friction May Distroy The Fix-Point Info, So We Need To use InvMass Again...

		if (vert_inv_mass[0] != 0.0f) { iter_position_cloth[indices[0]] += dx[0]; } 
		if (vert_inv_mass[1] != 0.0f) { iter_position_tet[indices[1]] += dx[1]; }
		
	}
}



inline void solve_self_collision_vv_per_collision_pair_template(
	const uint pair_idx, 
	PTR(Float3) substep_start_position, 
	PTR(Float3) input_position, 
	PTR(Float3) output_position, 
	
	PTR(float) sa_vert_mass_inv, 
	PTR(ATOMIC_FLAG) sa_vert_mutex,

	PTR(ProximityVV) self_collision_pair_vv,
	PTR(float) lambda_self_collision,
	PTR(float) lambda_self_collision_friction,

	const float substep_dt, 
	const bool use_atomic,
	const float thickness,
	const float stiffness_collision,
	const float stiffness_friction)
{
	const auto pair = self_collision_pair_vv[pair_idx];
	const Int2 indices = pair.get_indices();
	const Float3 vert_pos[2] = {
		input_position[indices[0]],
		input_position[indices[1]],
	};
	const Float3 start_vert_pos[2] = {
		substep_start_position[indices[0]],
		substep_start_position[indices[1]],
	};
	const float vert_inv_mass[2] = {
		sa_vert_mass_inv[indices[0]],
		sa_vert_mass_inv[indices[1]],
	};
	// const Float2 weight = pair.get_weight();
	const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	const Float3 diff = vert_pos[0] - vert_pos[1];
	const float proj = dot_vec(diff, normal);

	if (proj < thickness)
	{
		const float lambda_n = lambda_self_collision[pair_idx];
		float lambda_f = lambda_self_collision_friction[pair_idx];
		{
			//  To satisfy Coulomb's law that the frictional force should be limited by the normal force 
			//  we clamp the frictional Lagrange multiplier updates as follows
			lambda_f = min_scalar(stiffness_friction * lambda_f, lambda_n);
		}
		

		Float3 dx[2] = {Zero3, Zero3};
		const float alpha_tilde = 1.0f / (stiff * substep_dt * substep_dt);

		// Non-Elastic Collision
		{
			const float C_n = thickness - proj;

			float energy; Float3 gradient;
			get_collision_C_and_dCdx(C_n, normal, stiff, energy, gradient);

			const float delta_lambda = Constrains::Core::get_delta_xpbd_2(
				dx[0], dx[1], gradient, -gradient, vert_inv_mass[0], vert_inv_mass[1], 
				energy, lambda_n, alpha_tilde);

			// const Float3 gradient1 = normal ; //* weight[0];
			// const Float3 gradient2 = -gradient1;
			// const float delta_lambda = Constrains::Core::get_delta_xpbd_2(
			//     dx[0], dx[1], gradient1, gradient2, vert_inv_mass[0], vert_inv_mass[1], 
			//     C_n, lambda_n, alpha_tilde);
			
			// const float energy = 0.5f * stiff * C_n * C_n;
			// const Float3 force1 = stiff * C_n * weight[0] * normal;
			// const Float3 force2 = -force1;
			// const float delta_lambda = Constrains::Core::get_delta_xpbd_2(
			// 	dx[0], dx[1], -force1, -force2, vert_inv_mass[0], vert_inv_mass[1], 
			// 	energy, lambda_n, alpha_tilde);
			
			lambda_self_collision[pair_idx] += delta_lambda;
		}
		// Friction
		{				
			Float3 v1 = (vert_pos[0] - start_vert_pos[0]) / substep_dt;
			Float3 v2 = (vert_pos[1] - start_vert_pos[1]) / substep_dt;
			Float3 v_avg = 0.5f * (v1 + v2);
			const float friction_damping = 0.5f;
			dx[0] += friction_damping * substep_dt * (v_avg - v1);
			dx[1] += friction_damping * substep_dt * (v_avg - v2);
		}

		Constrains::Core::assemble_to_position(output_position, sa_vert_mutex, indices, dx, use_atomic);
		
	}
}
inline void solve_self_collision_vf_per_collision_pair_template(
	const uint pair_idx, 
	PTR(Float3) substep_start_position, 
	PTR(Float3) input_position, 
	PTR(Float3) output_position, 
	
	PTR(float) sa_vert_mass_inv, 
	PTR(ATOMIC_FLAG) sa_vert_mutex,

	PTR(ProximityVF) self_collision_pair_vf,
	PTR(float) lambda_self_collision,
	PTR(float) lambda_self_collision_friction,

	const float substep_dt, 
	const bool use_atomic,
	const float thickness,
	const float stiffness_collision,
	const float stiffness_friction)
{
	const auto pair = self_collision_pair_vf[pair_idx];
	const Int4 indices = pair.get_indices();
	const Float3 vert_pos[4] = {
		input_position[indices[0]],
		input_position[indices[1]],
		input_position[indices[2]],
		input_position[indices[3]],
	};
	// const Float3 start_vert_pos[4] = {
	// 	substep_start_position[indices[0]],
	// 	substep_start_position[indices[1]],
	// 	substep_start_position[indices[2]],
	// 	substep_start_position[indices[3]],
	// };
	const float vert_inv_mass[4] = {
		sa_vert_mass_inv[indices[0]],
		sa_vert_mass_inv[indices[1]],
		sa_vert_mass_inv[indices[2]],
		sa_vert_mass_inv[indices[3]],
	};

	// const Float3 normal = pair.get_normal();
	const float stiff = pair.get_stiff();
	// const Float3 bary = pair.get_face_weight();
	// const Float3 xs = bary[0] * vert_pos[1] + bary[1] * vert_pos[2] + bary[2] * vert_pos[3];
	// const Float3 diff = vert_pos[0] - xs;
	// const float proj = dot_vec(diff, normal);
	
	float distance; Float3 t; Float4 weight; const bool ignore_vert_projection_not_in_triangle = false;
	const bool is_collision = NarrowPhase::template_vf<ignore_vert_projection_not_in_triangle>(vert_pos, thickness, distance, t, weight);
	// const Float3 diff = t; 
	const Float3 bary = makeFloat3(weight[1], weight[2], weight[3]); const Float3 normal = t / distance;
	const float proj = distance;

	if (is_collision && proj < thickness)
	{
		const float lambda_n = lambda_self_collision[pair_idx];
		float lambda_f = lambda_self_collision_friction[pair_idx];
		{
			//  To satisfy Coulomb's law that the frictional force should be limited by the normal force 
			//  we clamp the frictional Lagrange multiplier updates as follows
			lambda_f = min_scalar(stiffness_friction * lambda_f, lambda_n);
		}
		

		Float3 dx[4] = {Zero3, Zero3, Zero3, Zero3};
		const float alpha_tilde = 1.0f / (stiff * substep_dt * substep_dt);

		// Non-Elastic Collision
		{
			const float C_n = thickness - proj;

			float energy; Float3 gradient;
			get_collision_C_and_dCdx(C_n, normal, stiff, energy, gradient);

			const float delta_lambda = Constrains::Core::get_delta_xpbd_4(
				dx[0], dx[1], dx[2], dx[3], 
				gradient, -bary[0] * gradient, -bary[1] * gradient, -bary[2] * gradient, 
				vert_inv_mass[0], vert_inv_mass[1], vert_inv_mass[2], vert_inv_mass[3], 
				energy, lambda_n, alpha_tilde);
			
			lambda_self_collision[pair_idx] += delta_lambda;
		}
		// Friction
		{
			// Float3 vel_p =  (vert_pos[0] - start_vert_pos[0]) / substep_dt;
			// Float3 vel_f1 = (vert_pos[1] - start_vert_pos[1]) / substep_dt;
			// Float3 vel_f2 = (vert_pos[2] - start_vert_pos[2]) / substep_dt;
			// Float3 vel_f3 = (vert_pos[3] - start_vert_pos[3]) / substep_dt;
			// Float3 vel_f = 0.3333f * (vel_f1 + vel_f2 + vel_f3);
			// Float3 v_avg = 0.5f * (vel_p + vel_f);
			// const float friction_damping = 0.5f;
			// dx[0] += friction_damping * substep_dt * (v_avg - vel_p);
			// dx[1] += friction_damping * substep_dt * bary[0] * (v_avg - vel_f);
			// dx[2] += friction_damping * substep_dt * bary[1] * (v_avg - vel_f);
			// dx[3] += friction_damping * substep_dt * bary[2] * (v_avg - vel_f);
		}

		Constrains::Core::assemble_to_position(output_position, sa_vert_mutex, indices, dx, use_atomic);
		
	}
}


// Obstacle-Collision: VF Only
inline void solve_obstacle_collision_vf_template_cloth(
	const uint i, 
	PTR(Float3) iter_position_cloth, 
	PTR(Float3) iter_position_tet, 
	PTR(Float3) sa_obstacle_start_position,
	PTR(Float3) sa_obstacle_velocity,

	PTR(Float3) output_position_cloth, 
	PTR(Float3) output_position_tet,

	PTR(Float3) substep_start_position_cloth, 
	PTR(Float3) substep_start_position_tet,

	PTR(uint) sa_surface_verts, 
	PTR(float) sa_vert_mass_inv_cloth, 
	PTR(float) sa_vert_mass_inv_tet, 
	PTR(ATOMIC_FLAG) sa_vert_mutex_cloth,
	PTR(ATOMIC_FLAG) sa_vert_mutex_tet,

	PTR(uint) vert_VV_num_narrow_phase, 
	PTR(uint) vert_VV_prefix_narrow_phase, 
	PTR(uint) vert_adj_elements,
	PTR(ProximityVF) narrow_phase_list_vf,

	PTR(float) lambda_obstacle_collision_n,
	PTR(float) lambda_obstacle_collision_f,

	const uint max_vv_per_vert_narrow_obstacle_collision,
	const float thickness, 
	const float substep_dt,
	const float stiffness_collision,
	const float stiffness_friction,
	const uint num_verts_cloth)
{
	const uint vid = i ;
	const float vert_mass_inv = sa_vert_mass_inv_cloth[vid];
	const float is_fixed = vert_mass_inv == 0.0f;
	if (is_fixed) return;

	const uint num_adj = vert_VV_num_narrow_phase[vid];
    if (num_adj != 0) 
	{
		const Float3 start_pos = substep_start_position_cloth[vid];
		Float3 vert_pos = iter_position_cloth[vid];
		
		const uint start_idx = vert_VV_prefix_narrow_phase[vid];
		for (uint j = 0; j < num_adj; j++) 
		{
			const uint pair_idx = vert_adj_elements[start_idx + j];
			const auto pair = narrow_phase_list_vf[pair_idx];
			
			const Int4 indices = pair.get_indices();

			Float3 obs_vert_pos[3] = {
				sa_obstacle_start_position[indices[1]],
				sa_obstacle_start_position[indices[2]],
				sa_obstacle_start_position[indices[3]]
			};
			
			const Float3 normal = pair.get_normal();
			const float stiff = pair.get_stiff();
			const Float3 bary = pair.get_face_weight();
			const Float3 xs = bary[0] * obs_vert_pos[0] + bary[1] * obs_vert_pos[1] + bary[2] * obs_vert_pos[2];
			const Float3 diff = vert_pos - xs;
			const float proj = dot_vec(diff, normal);
			const bool is_collision = proj < thickness;

			if (is_collision)
			{
				// const float lambda_n = lambda_obstacle_collision_n[pair_idx];
				const float lambda_n = 0.0f;
				// const float lambda_f = lambda_obstacle_collision_f[pair_idx];
				
				Float3 dx = Zero3;
				const float alpha_tilde_n = 1.0f / (stiff * substep_dt * substep_dt);

				// Non-Elastic Collision
				{
					const float C_n = thickness - proj;

					float energy; Float3 gradient;
					get_collision_C_and_dCdx(C_n, normal, stiff, energy, gradient);

					// float energy = C_n;
					// Float3 gradient = -normal;
					// alpha_tilde = 0.0f; 

					const float delta_lambda_n = Constrains::Core::get_delta_xpbd_1(
						dx, gradient, vert_mass_inv, 
						energy, lambda_n, alpha_tilde_n);
					
					lambda_obstacle_collision_n[pair_idx] += delta_lambda_n;

					// Friction
					{
						Float3 v1 = (vert_pos - start_pos) / substep_dt;
						Float3 v2 = 0.33333f * (sa_obstacle_velocity[indices[1]] + sa_obstacle_velocity[indices[2]] + sa_obstacle_velocity[indices[3]]);	
						Float3 v_avg = 0.5f * (v1 + v2);
						const float friction_damping = stiffness_friction;
						dx += 2.0f * friction_damping * substep_dt * (v_avg - v1);

						// Friction
						{			
							// Float3 v1 = (vert_pos - start_pos) / substep_dt;
							// Float3 v2 = 0.33333f * (sa_obstacle_velocity[indices[1]] + sa_obstacle_velocity[indices[2]] + sa_obstacle_velocity[indices[3]]);
							// Float3 relative_vel = v1 - v2;
							// Float3 vel_proj = relative_vel - dot_vec(normal, relative_vel) * normal; // project_vec(relative_vel, normal);

							// float delta_lambda_f = max_scalar(0.5f * delta_lambda_n, 0.0 - length_vec(vel_proj) * substep_dt);							
							// dx -= vert_mass_inv * delta_lambda_f * normalize_vec(vel_proj);
							// lambda_obstacle_collision_f[pair_idx] += delta_lambda_f;

							// const float mu_kinetic = 0.5f;
							// const float mu_static = 0.6f;
							// const THREAD float& d = C_f; // (thickness - proj);
							// if (C_f < d * mu_static)
							// {
							// 	dx -= vel_proj;
							// }
							// else 
							// {
							// 	dx -= vel_proj * min_scalar(d * mu_kinetic, 1.0f);
							// }
							
							
							// const float alpha_tilde_f = 1.0f / (mu_kinetic * substep_dt * substep_dt);

							// 
							// if (C_f < d * mu_static)
							// {
							// 	// Static Friction
							// 	C_f = C_f * min_scalar(d * mu_kinetic, 1.0f);
							// }
							// Float3 gradient_f = -normalize_vec(vel_proj);
							
							// const float denom = 
							// 	vert_mass_inv * length_squared_vec(gradient_f);
							// float delta_lambda_f = (-C_f) / (denom);
							// // const float denom = 
							// // 	vert_mass_inv * length_squared_vec(gradient_f) + alpha_tilde_f;
							// // float delta_lambda_f = (-C_f - alpha_tilde_f * lambda_f) / (denom);

							// // delta_lambda_f = max_scalar(mu_kinetic * delta_lambda_n, delta_lambda_f);

							

							// if (denom != 0.f)
							// {
							// 	const Float3 dx_f = delta_lambda_f * vert_mass_inv * gradient_f;
							// 	dx += dx_f;
							// }			
						}
					}
				}
				
				vert_pos += dx;
			}
		}

		output_position_cloth[vid] = vert_pos;
	}
}
inline void solve_obstacle_collision_vf_template_tet(
	const uint surface_id, 
	PTR(Float3) iter_position_cloth, 
	PTR(Float3) iter_position_tet, 
	PTR(Float3) sa_obstacle_start_position,
	PTR(Float3) sa_obstacle_velocity,

	PTR(Float3) output_position_cloth, 
	PTR(Float3) output_position_tet,

	PTR(Float3) substep_start_position_cloth, 
	PTR(Float3) substep_start_position_tet,

	PTR(uint) sa_surface_verts, 
	PTR(float) sa_vert_mass_inv_cloth, 
	PTR(float) sa_vert_mass_inv_tet, 
	PTR(ATOMIC_FLAG) sa_vert_mutex_cloth,
	PTR(ATOMIC_FLAG) sa_vert_mutex_tet,

	PTR(uint) vert_VV_num_narrow_phase, 
	PTR(uint) vert_VV_prefix_narrow_phase, 
	PTR(uint) vert_adj_elements,
	PTR(ProximityVF) narrow_phase_list_vf,

	PTR(float) lambda_obstacle_collision,
	PTR(float) lambda_obstacle_collision_friction,

	const uint max_vv_per_vert_narrow_obstacle_collision,
	const float thickness, 
	const float substep_dt,
	const float stiffness_collision,
	const float stiffness_friction,
	const uint num_verts_cloth)
{
	const uint vid = sa_surface_verts[surface_id];
	const float vert_mass_inv = sa_vert_mass_inv_tet[vid];
	const float is_fixed = vert_mass_inv == 0.0f;
	if (is_fixed) return;

	// const uint num_adj = vert_VV_num_narrow_phase[num_verts_cloth + surface_id];
	const uint num_adj = vert_VV_num_narrow_phase[surface_id];
    if (num_adj != 0) 
	{
		const Float3 start_pos = substep_start_position_tet[vid];
		Float3 vert_pos = iter_position_tet[vid];
		
		// const uint start_idx = vert_VV_prefix_narrow_phase[num_verts_cloth + surface_id];
		const uint start_idx = vert_VV_prefix_narrow_phase[surface_id];
		for (uint j = 0; j < num_adj; j++) 
		{
			const uint pair_idx = vert_adj_elements[start_idx + j];
			const auto pair = narrow_phase_list_vf[pair_idx];
			
			const Int4 indices = pair.get_indices();

			Float3 obs_vert_pos[3] = {
				sa_obstacle_start_position[indices[1]],
				sa_obstacle_start_position[indices[2]],
				sa_obstacle_start_position[indices[3]]
			};
			
			const Float3 normal = pair.get_normal();
			const float stiff = pair.get_stiff();
			const Float3 bary = pair.get_face_weight();
			const Float3 xs = bary[0] * obs_vert_pos[0] + bary[1] * obs_vert_pos[1] + bary[2] * obs_vert_pos[2];
			const Float3 diff = vert_pos - xs;
			const float proj = dot_vec(diff, normal);
			const bool is_collision = proj < thickness;

			if (is_collision)
			{
				const float lambda_n = lambda_obstacle_collision[pair_idx];
				
				Float3 dx = Zero3;
				const float alpha_tilde = 1.0f / (stiff * substep_dt * substep_dt);

				// Non-Elastic Collision
				{
					const float C_n = thickness - proj;

					float energy; Float3 gradient;
					get_collision_C_and_dCdx(C_n, normal, stiff, energy, gradient);

					const float delta_lambda = Constrains::Core::get_delta_xpbd_1(
						dx, gradient, vert_mass_inv, 
						energy, lambda_n, alpha_tilde);
					
					lambda_obstacle_collision[pair_idx] += delta_lambda;
				}
				// Friction
				{
					// const float friction_damping = 0.5f;
					// Float3 dx_h = (vert_pos - start_pos);
					// dx -= friction_damping * dx_h;
					Float3 v1 = (vert_pos - start_pos) / substep_dt;
					Float3 v2 = 0.33333f * (sa_obstacle_velocity[indices[1]] + sa_obstacle_velocity[indices[2]] + sa_obstacle_velocity[indices[3]]);	
					Float3 v_avg = 0.5f * (v1 + v2);
					const float friction_damping = stiffness_friction;
					dx += 2.0f * friction_damping * substep_dt * (v_avg - v1);

				}

				vert_pos += dx;
			}
		}

		output_position_tet[vid] = vert_pos;
	}
}



}


// VBD Evaluate
namespace Constrains
{

namespace VBD 
{

template <bool compute_energy, bool compute_gradient, bool compute_hessian>
inline void local_core_bending_quadratic(
    THREAD float* energy, THREAD Float3* force, THREAD Float3x3* hessian,
    const THREAD Float3 vert_pos[3],
    CREF(Float4x4) m_Q,   
    const float stiffness
)
{
    //
    // A quadratic bending model for inextensible surfaces
    //
    CONSTIF (compute_gradient)
    {
        for (uint ii = 0; ii < 4; ii++) 
        {
            for (uint jj = 0; jj < 4; jj++) 
            {
                force[ii] -= get(m_Q, ii, jj) * vert_pos[jj]; // -Qx
            }
            force[ii] = stiffness * force[ii];
        }
    }

    /// hessian
    //  0   1   2   3
    // t1   4   5   6
    // t2  t5   7   8
    // t3  t6  t8   9 
    CONSTIF (compute_hessian)
    {
        // uint idx = 0;
        // for (uint i = 0; i < 4; i++) 
        // {
        //     for (uint j = i; j < 4; j++)
        //     {
        //         hessian[idx] = stiffness * get(m_Q, j, i) * Identity3x3; 
        //         idx++;
        //     }
        // }
        hessian[0] = stiffness * get(m_Q, 0, 0) * Identity3x3;
        hessian[1] = stiffness * get(m_Q, 1, 1) * Identity3x3;
        hessian[2] = stiffness * get(m_Q, 2, 2) * Identity3x3;
        hessian[3] = stiffness * get(m_Q, 3, 3) * Identity3x3;
    }
}


#define Zero4x3 make<Float4x3>(make<Float3>(0.f, 0.f, 0.f),  \
                               make<Float3>(0.f, 0.f, 0.f),  \
                               make<Float3>(0.f, 0.f, 0.f),  \
                               make<Float3>(0.f ,0.f, 0.f))

inline Float4x3 makeHf(CREF(Float3) force, CREF(Float3x3) hessian)
{
	return makeFloat4x3(get_column(hessian, 0), get_column(hessian, 1), get_column(hessian, 2), force);
}
inline void extractHf(CREF(Float4x3) Hf, TREF(Float3) force, TREF(Float3x3) hessian)
{
	hessian = makeFloat3x3(get(Hf, 0), get(Hf, 1), get(Hf, 2));
	force = get(Hf, 3);
}

inline Float4x3 compute_inertia(
	const uint vid, PTR(Float3) sa_iter_position, 
	PTR(Float3) sa_iter_start_position, PTR(Float3) sa_vert_velocity,
	PTR(uchar) sa_is_fixed, PTR(float) sa_vert_mass, PTR(SceneParams) scene_params,
	const float substep_dt
	)
{
	const Float3 gravity = scene_params->gravity;
	// const Float3 floor = get_scene_params().floor;
	// const float h = get_scene_params().implicit_dt;
	const float h = substep_dt;
	// const bool use_floor = get_scene_params().use_floor;
	const float h_2_inv = 1.f / (h * h);

	Float3 x_k = sa_iter_position[vid];
	Float3 x_0 = sa_iter_start_position[vid];
	Float3 v_0 = sa_vert_velocity[vid];

	bool is_fixed = sa_is_fixed[vid];
	float mass = sa_vert_mass[vid];
	Float3x3 mat = Identity3x3 * mass * h_2_inv; // A = M / h^2

	Float3 outer_force = Zero3;
	outer_force += mass * gravity; // G = mg

	// if (use_floor)
	// {
	//     CollisionForce::compute_force_boundary(floor, x_k, outer_force, mat);
	// }

	if (is_fixed)
	{
		mat += Identity3x3 * float(1E9);
		// sa_iter_position[vid] = x_0;
	}

	return makeHf(
		-mass * h_2_inv * (x_k - x_0 - v_0 * h) + outer_force,
		mat
	);
};

inline Float4x3 compute_stretch_fem(
	const uint vid, PTR(Float3) sa_iter_position, 
	PTR(uint) sa_vert_adj_faces_csr, 
	PTR(Int3) sa_faces, PTR(Float2x2) sa_inv_duv, PTR(float) sa_face_area,
	const float stiffness_stretch
	)
{
	const uint curr_prefix = sa_vert_adj_faces_csr[vid];
	const uint next_prefix = sa_vert_adj_faces_csr[vid + 1];
	const uint num_adj = next_prefix - curr_prefix;

	Float4x3 hf = Zero4x3;
	for (uint j = 0; j < num_adj; j++)
	{
		// const uint adj_fid = sa_vert_adj_faces_csr[curr_prefix + j];
		// Int3 face = sa_faces[adj_fid];

		// Float3 vert_pos[3] = {
		// 	sa_iter_position[face[0]],
		// 	sa_iter_position[face[1]],
		// 	sa_iter_position[face[2]],
		// };
		// Float3 force[3] = {Zero3, Zero3, Zero3};
		// Float3x3 hessian[3] = {Zero3x3, Zero3x3, Zero3x3};
		// float energy = 0.0f;

		// const Float2x2 inv_duv = sa_inv_duv[adj_fid];
		// const float face_area = sa_face_area[adj_fid];
		// local_stretch_baraff_witkin<false, true, true>(&energy, force, hessian, vert_pos, inv_duv, face_area, stiffness_stretch);           

		// const uint offset = vid == face[0] ? 0 : vid == face[1] ? 1 : vid == face[2] ? 2 : -1u;
		// cgB += force[offset];
		// cgA += hessian[offset];
		// hf += makeHf(force[offset], hessian[offset]);
	}
	return hf;
};

inline Float4x3 compute_stretch_mass_spring(
	const uint vid, PTR(Float3) sa_iter_position, 
	PTR(uint) sa_vert_adj_edges_csr, 
	PTR(Int2) sa_edges, 
	PTR(float) sa_rest_length,
	const float stiffness_stretch
	) 
{
	const uint curr_prefix = sa_vert_adj_edges_csr[vid];
	const uint next_prefix = sa_vert_adj_edges_csr[vid + 1];
	const uint num_adj = next_prefix - curr_prefix;

	Float4x3 hf = Zero4x3;
	for (uint j = 0; j < num_adj; j++)
	{
		const uint adj_eid = sa_vert_adj_edges_csr[curr_prefix + j];
		Int2 edge = sa_edges[adj_eid];

		Float3 vert_pos[2] = {
			sa_iter_position[edge[0]],
			sa_iter_position[edge[1]],
		};
		Float3 force[2] = {Zero3, Zero3};
		Float3x3 He = Zero3x3;
 
		{
			const float L = sa_rest_length[adj_eid];
			const float stiffness_stretch_spring = stiffness_stretch;

			Float3 diff = vert_pos[1] - vert_pos[0];
			float l = max_scalar(length_vec(diff), Epsilon);
			float l0 = L;
			float C = l - l0;
			// if (C > Epsilon)
			{
				Float3 dir = diff / l;
				force[0] = stiffness_stretch_spring * dir * C;
				force[1] = -force[0];

				{
					Float3x3 xxT = outer_product(diff, diff);
					float x_inv = 1.f / l;
					float x_squared_inv = x_inv * x_inv;
					He = stiffness_stretch_spring * x_squared_inv * xxT + stiffness_stretch_spring * max_scalar(1 - L * x_inv, 0.0f) * (Identity3x3 - x_squared_inv * xxT) ;
				}
			}
		}
			
		const uint offset = vid == edge[0] ? 0 : vid == edge[1] ? 1 : -1u;
		hf += Constrains::VBD::makeHf(force[offset], He);
	}
	return hf;
};

inline Float4x3 compute_bending_quadratic(
	const uint vid, PTR(Float3) sa_iter_position,
	PTR(uint) sa_vert_adj_bending_edges_csr,
	PTR(Int4) sa_bending_edges, PTR(Float4x4) sa_bending_edge_Q, const float stiffness_quadratic_bending
	)
{
	const uint curr_prefix = sa_vert_adj_bending_edges_csr[vid];
	const uint next_prefix = sa_vert_adj_bending_edges_csr[vid + 1];
	const uint num_adj_edges = next_prefix - curr_prefix;

	Float4x3 hf = Zero4x3;
	for (uint j = 0; j < num_adj_edges; j++)
	{
		const uint adj_eid = sa_vert_adj_bending_edges_csr[curr_prefix + j];            
		Int4 edge = sa_bending_edges[adj_eid];

		Float3 vert_pos[4] = {
			sa_iter_position[edge[0]],
			sa_iter_position[edge[1]],
			sa_iter_position[edge[2]],
			sa_iter_position[edge[3]],
		};
		
		
		Float4x4 Q = sa_bending_edge_Q[adj_eid];
		const uint offset = vid == edge[0] ? 0 : vid == edge[1] ? 1 : vid == edge[2] ? 2 : vid == edge[3] ? 3 : -1u;

		Float3 force = Zero3;
		Float3x3 hessian = Zero3x3;
		for (uint jj = 0; jj < 4; jj++) 
		{
			force -= get(Q, offset, jj) * vert_pos[jj]; // -Qx
		}
		force *= stiffness_quadratic_bending;
		hessian = stiffness_quadratic_bending * get(Q, offset, offset) * Identity3x3;
		hf += makeHf(force, hessian);

		// Float3 force[4] = {Zero3, Zero3, Zero3, Zero3};
		// Float3x3 hessian[4] = {Zero3x3, Zero3x3, Zero3x3, Zero3x3};;
		// local_core_bending_quadratic<false, true, true>(
		// 	nullptr, force, hessian, vert_pos, Q, stiffness_quadratic_bending);
		// hf += makeHf(force[offset], hessian[offset]);
	}
	return hf;
};

inline Float4x3 compute_ground_collision(
	const uint vid,
	PTR(Float3) sa_iter_position, const float stiffness_collision,
	const float thickness_vv_obstacle, PTR(SceneParams) scene_param
	)
{
	const Float3 normal = makeFloat3(0, 1, 0);
	if (!scene_param->use_floor) return makeHf(Zero3, Zero3x3);
	
	const Float3 floor = scene_param->floor;
	Float3 vert_pos = sa_iter_position[vid];
	
	const float stiff = stiffness_collision;
	const Float3 diff = vert_pos - floor;
	const float proj = dot_vec(diff, normal);
	const float thickness = thickness_vv_obstacle;
	const bool is_collision = proj < thickness;

	Float4x3 hf = Zero4x3;
	if (is_collision)
	{
		const float C_n = thickness - proj;
		Float3 force = stiff * C_n * normal;
		Float3x3 hessian = outer_product(normal, stiff * normal);

		// cgB += force;
		// cgA += hessian;
		hf = makeHf(force, hessian);
	}
	return hf;
};
inline Float4x3 compute_obstacle_collision(
	const uint vid, PTR(Float3) sa_iter_position, PTR(Float3) sa_obstacle_substep_position,
	PTR(uint) vert_VV_prefix_narrow_phase, PTR(uint) vert_VV_num_narrow_phase, PTR(uint) vert_adj_elements,
	PTR(ProximityVF) narrow_phase_list_pair_vf, const float thickness_vv_obstacle, const float stiffness_collision
	)
{
	const uint curr_prefix = vert_VV_prefix_narrow_phase[vid];
	// const uint next_prefix = vert_VV_prefix_narrow_phase[vid + 1];
	const uint num_adj = vert_VV_num_narrow_phase[vid];

	Float4x3 hf = Zero4x3;
	for (uint j = 0; j < num_adj; j++)
	{
		const uint adj_pairidx = vert_adj_elements[curr_prefix + j];
		const auto pair = narrow_phase_list_pair_vf[adj_pairidx];
		
		Int4 indices = pair.get_indices();
		Float3 vert_pos[4] = {
			sa_iter_position[indices[0]],
			sa_obstacle_substep_position[indices[1]],
			sa_obstacle_substep_position[indices[2]],
			sa_obstacle_substep_position[indices[3]],
		};

		const Float3 normal = pair.get_normal();
		const float stiff = pair.get_stiff();
		// const float stiff = stiffness_collision;
		const Float3 bary = pair.get_face_weight();
		const Float3 xs = bary[0] * vert_pos[1] + bary[1] * vert_pos[2] + bary[2] * vert_pos[3];
		const Float3 diff = vert_pos[0] - xs;
		const float proj = dot_vec(diff, normal);
		const float thickness = thickness_vv_obstacle;
		const bool is_collision = proj < thickness;

		if (is_collision)
		{
			const float C_n = thickness - proj;
			Float3 force = stiff * C_n * normal;
			Float3x3 hessian = outer_product(normal, stiff * normal);

			// cgB += force;
			// cgA += hessian;
			hf += makeHf(force, hessian);
		}
	}
	return hf;
};
inline Float4x3 compute_self_collision(
	const uint vid, PTR(Float3) sa_iter_position, 
	PTR(uint) vert_VV_prefix_narrow_phase, PTR(uint) vert_VV_num_narrow_phase, PTR(uint) vert_adj_elements,
	PTR(ProximityVV) narrow_phase_list_pair_vv, const float thickness_vv_cloth, const float stiffness_collision
	)
{
	const uint curr_prefix = vert_VV_prefix_narrow_phase[vid];
	// const uint next_prefix = vert_VV_prefix_narrow_phase[vid + 1];
	const uint num_adj = vert_VV_num_narrow_phase[vid];

	Float4x3 hf = Zero4x3;
	for (uint j = 0; j < num_adj; j++)
	{
		const uint adj_pairidx = vert_adj_elements[curr_prefix + j];
		const auto pair = narrow_phase_list_pair_vv[adj_pairidx];
		
		Int2 indices = pair.get_indices();
		Float3 vert_pos[2] = {
			sa_iter_position[indices[0]], 
			sa_iter_position[indices[1]], 
		};

		const Float3 normal = pair.get_normal();
		// const float stiff = stiffness_collision;
		const float stiff = pair.get_stiff();

		const Float3 diff = vert_pos[0] - vert_pos[1];
		const float proj = dot_vec(diff, normal);
		const float thickness = thickness_vv_cloth;
		const bool is_collision = proj < thickness;

		if (is_collision)
		{
			const float C_n = thickness - proj;
			Float3 force = vid == indices[0] ? stiff * C_n * normal : -stiff * C_n * normal;
			Float3x3 hessian = outer_product(normal, stiff * normal);

			// cgB += force;
			// cgA += hessian;
			hf += makeHf(force, hessian);
		}
	}
	return hf;
};


}

}