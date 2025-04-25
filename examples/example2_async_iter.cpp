#include <iostream>
#include <tbb/tbb.h>
#include "shared_array.h"
#include "command_list.h"
#include "mesh_reader.h"
#include "scene_params.h"
#include "fem_energy.h"
#include "xpbd_constraints.h"

struct BasicMeshData
{
    uint num_verts;
    uint num_faces;
    uint num_edges;
    uint num_bending_edges;

    SharedArray<Float3> sa_rest_x;
    SharedArray<Float3> sa_rest_v;

    SharedArray<Float3> sa_x_frame_start;
    SharedArray<Float3> sa_v_frame_start;
    SharedArray<Float3> sa_x_frame_end;
    SharedArray<Float3> sa_v_frame_end;

    SharedArray<Int3> sa_faces;
    SharedArray<Int2> sa_edges;
    SharedArray<Int4> sa_bending_edges;
    
    SharedArray<float> sa_vert_mass;
    SharedArray<float> sa_vert_mass_inv;
    SharedArray<uchar> sa_is_fixed;

    SharedArray<uint> sa_vert_adj_faces; std::vector< std::vector<uint> > vert_adj_faces;
    SharedArray<uint> sa_vert_adj_edges; std::vector< std::vector<uint> > vert_adj_edges;
    SharedArray<uint> sa_vert_adj_bending_edges; std::vector< std::vector<uint> > vert_adj_bending_edges;
    
    // SharedArray<uint> sa_edge_adj_faces;
    // SharedArray<uint> sa_bending_edge_adj_faces;
};

struct XpbdData
{
    SharedArray<Float3> sa_x_tilde;
    SharedArray<Float3> sa_x;
    SharedArray<Float3> sa_v;
    SharedArray<Float3> sa_v_start;
    SharedArray<Float3> sa_x_start; // For calculating velocity

    SharedArray<Int2> sa_merged_edges; SharedArray<float> sa_merged_edges_rest_length;
    SharedArray<Int4> sa_merged_bending_edges; SharedArray<float> sa_merged_bending_edges_rest_angle;

    uint num_clusters_stretch_mass_spring = 0;
    SharedArray<uint> clusterd_constraint_stretch_mass_spring; 
    SharedArray<uint> prefix_stretch_mass_spring;
    SharedArray<float> sa_lambda_stretch_mass_spring;

    uint num_clusters_bending = 0;
    SharedArray<uint> clusterd_constraint_bending; 
    SharedArray<uint> prefix_bending; 
    SharedArray<float> sa_lambda_bending;

#if false
    const bool check1()
    {
        return !(
            sa_x_tilde.is_empty() || 
            sa_x.is_empty() || 
            sa_v.is_empty() || 
            sa_v_start.is_empty() || 
            sa_x_start.is_empty());
    }
    const bool check2()
    {
        return !(
            sa_merged_edges.is_empty() || 
            sa_merged_edges_rest_length.is_empty() || 
            sa_merged_bending_edges.is_empty() || 
            sa_merged_bending_edges_rest_angle.is_empty()
        );
    }
    const bool check3()
    {
        return !(
            clusterd_constraint_stretch_mass_spring.is_empty() || 
            prefix_stretch_mass_spring.is_empty() || 
            sa_lambda_stretch_mass_spring.is_empty()
        );
    }
    const bool check4()
    {
        return !(
            clusterd_constraint_bending.is_empty() || 
            prefix_bending.is_empty() || 
            sa_lambda_bending.is_empty()
        );
    }

    void check()
    {
        if (!(
            check1() && check2() && check3() && check4() 
        ))
        {
            fast_format_err("exist empty buffer");
        }
    }
#endif
};


void init_mesh(BasicMeshData* mesh_data)
{
    std::string model_name = "square8K.obj";
    Float3 transform = make<Float3>(0.0f);
    Float3 rotation = make<Float3>(0.0f * Pi);
    Float3 scale = makeFloat3(1.0f);


    TriangleMeshData input_mesh;
    bool second_read = SimMesh::read_mesh_file(model_name, input_mesh, true);

    std::string obj_name = model_name;
    {
        std::filesystem::path path(obj_name);
        obj_name = path.stem().string();
    }

    const uint num_verts = input_mesh.model_positions.size();
    const uint num_faces = input_mesh.faces.size();
    const uint num_edges = input_mesh.edges.size();
    const uint num_bending_edges = input_mesh.bending_edges.size();

    // Constant scalar
    {
        mesh_data->num_verts = num_verts; mesh_data->sa_rest_x.upload(input_mesh.model_positions); 
        mesh_data->num_faces = num_faces; mesh_data->sa_faces.upload(input_mesh.faces); 
        mesh_data->num_edges = num_edges; mesh_data->sa_edges.upload(input_mesh.edges); 
        mesh_data->num_bending_edges = num_bending_edges; mesh_data->sa_bending_edges.upload(input_mesh.bending_edges); 
    }
    // Init vert info
    {
        // Set rest position & velocity
        {
            parallel_for(0, num_verts, [&](const uint vid)
            {
                Float3 model_position = mesh_data->sa_rest_x[vid];
                Float4x4 model_matrix = make_model_matrix(transform, rotation, scale);
                Float3 world_position = affine_position(model_matrix, model_position);
                mesh_data->sa_rest_x[vid] = world_position;
            });

            mesh_data->sa_rest_v.resize(num_verts); 
            mesh_data->sa_rest_v.set_zero();
        }

        // Set fixed-points
        {
            mesh_data->sa_is_fixed.resize(num_verts);

            AABB local_aabb = parallel_for_and_reduce_sum<AABB>(0, mesh_data->sa_rest_x.size(), [&](const uint vid)
            {
                return AABB(mesh_data->sa_rest_x[vid]);
            });

            Float3 pos_min = local_aabb.min_pos;
            Float3 pos_max = local_aabb.max_pos;
            Float3 pos_dim = local_aabb.range();
            Float3 pos_dim_inv = local_aabb.range_inv();

            parallel_for(0, mesh_data->sa_rest_x.size(), [&](const uint vid)
            {
                Float3 orig_pos = mesh_data->sa_rest_x[vid];
                Float3 norm_pos = (orig_pos - pos_min) * pos_dim_inv;
                
                bool is_fixed = false;
                // is_fixed = norm_pos.y > 0.9f;
                // is_fixed = norm_pos.z < 0.5;
                // is_fixed = (norm_pos.x > 0.97f || norm_pos.x < 0.03f ) ;
                // is_fixed = (norm_pos.x > 0.999f || norm_pos.x < 0.001f ) ;
                is_fixed = norm_pos.z < 0.01f && (norm_pos.x > 0.99f || norm_pos.x < 0.01f ) ;
                // is_fixed = norm_pos.z < 0.001f && (norm_pos.x > 0.999f || norm_pos.x < 0.001f ) ;
                // is_fixed = norm_pos.z < 0.001f && (norm_pos.x < 0.001f) ;
                mesh_data->sa_is_fixed[vid] = is_fixed;
            });
        }

        // Set vert mass
        {
            mesh_data->sa_vert_mass.resize(num_verts); 
            mesh_data->sa_vert_mass_inv.resize(num_verts);

            const float defulat_density = 0.01f;
            const float defulat_mass = defulat_density * get_scene_params().default_mass;
            parallel_for(0, num_verts, [&](const uint vid)
            {
                bool is_fixed = mesh_data->sa_is_fixed[vid] != 0;
                mesh_data->sa_vert_mass[vid] = is_fixed ? 1e12 : (defulat_mass);
                mesh_data->sa_vert_mass_inv[vid] = is_fixed ? 0.0f : 1.0f / (defulat_mass);
            });
        }

    }
    // Init adjacent list
    {
        mesh_data->vert_adj_faces.resize(num_verts);
        mesh_data->vert_adj_edges.resize(num_verts);
        mesh_data->vert_adj_bending_edges.resize(num_verts);

        for (uint eid = 0; eid < num_faces; eid++)
        {
            auto edge = mesh_data->sa_faces[eid];
            for (uint j = 0; j < 3; j++)
                mesh_data->vert_adj_faces[edge[j]].push_back(eid);
        } 
        mesh_data->sa_vert_adj_faces.upload_2d_csr(mesh_data->vert_adj_faces); 

        for (uint eid = 0; eid < num_edges; eid++)
        {
            auto edge = mesh_data->sa_edges[eid];
            for (uint j = 0; j < 2; j++)
                mesh_data->vert_adj_edges[edge[j]].push_back(eid);
        } 
        mesh_data->sa_vert_adj_edges.upload_2d_csr(mesh_data->vert_adj_edges);

        for (uint eid = 0; eid < num_bending_edges; eid++)
        {
            auto edge = mesh_data->sa_bending_edges[eid];
            for (uint j = 0; j < 4; j++)
                mesh_data->vert_adj_bending_edges[edge[j]].push_back(eid);
        }  
        mesh_data->sa_vert_adj_bending_edges.upload_2d_csr(mesh_data->vert_adj_bending_edges);
    }

    // Init vert status
    {
        mesh_data->sa_x_frame_start.resize(num_verts); mesh_data->sa_x_frame_start = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_start.resize(num_verts); mesh_data->sa_v_frame_start = mesh_data->sa_rest_v;
        mesh_data->sa_x_frame_end.resize(num_verts); mesh_data->sa_x_frame_end = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_end.resize(num_verts); mesh_data->sa_v_frame_end = mesh_data->sa_rest_v;
    }
}

class XpbdSolver
{
public:
    XpbdSolver() { fast_format("Init for XPBD Solver"); }
    ~XpbdSolver() { fast_format("Destroy XPBD Solver"); }

    // TODO: Replace to shared_ptr
    void get_data_pointer(XpbdData* xpbd_ptr, BasicMeshData* mesh_ptr) 
    {
        xpbd_data = xpbd_ptr;
        mesh_data = mesh_ptr;
    }
    void init_xpbd_system();
    void init_simulation_params();

public:    
    void physics_step();

private:
    void collision_detection();
    void predict_position();
    void update_velocity();
    void reset_constrains();
    void reset_collision_constrains();

private:
    void solve_constraints();
    void solve_constraint_stretch_spring(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);
    void solve_constraint_bending(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);

private:
    XpbdData* xpbd_data;
    BasicMeshData* mesh_data;
};

void XpbdSolver::init_xpbd_system()
{
    xpbd_data->sa_x_tilde.resize(mesh_data->num_verts); 
    xpbd_data->sa_x.resize(mesh_data->num_verts);
    xpbd_data->sa_v.resize(mesh_data->num_verts); xpbd_data->sa_v = mesh_data->sa_rest_v;
    xpbd_data->sa_v_start.resize(mesh_data->num_verts); xpbd_data->sa_v_start = mesh_data->sa_rest_v;
    xpbd_data->sa_x_start.resize(mesh_data->num_verts);

    // Graph Coloring
    std::vector< std::vector<uint> > tmp_clusterd_constraint_stretch_mass_spring;
    std::vector< std::vector<uint> > tmp_clusterd_constraint_bending;

    {
        auto fn_graph_coloring_sequenced_constraint = [](const uint num_elements, const std::string& constraint_name, 
            std::vector< std::vector<uint> > & clusterd_constraint, 
            const std::vector< std::vector<uint> > & vert_adj_elements, const auto& element_indices, const uint nv)
        { 
            std::vector< bool > marked_constrains(num_elements, false);
            uint total_marked_count = 0;
        
        
            const uint color_threashold = 2000;
            std::vector<uint> rest_cluster;
        
            //
            // while there exist unmarked constraints
            //     create new set S
            //     clear all particle marks
            //     for all unmarked constraints C
            //      if no adjacent particle is marked
            //          add C to S
            //          mark C
            //          mark all adjacent particles
            //
            
            const bool merge_small_cluster = false;
        
            while (true) 
            {
                std::vector<uint> current_cluster;
                std::vector<bool> current_marked(marked_constrains);
                for (uint id = 0; id < num_elements; id++) 
                {
                    if (current_marked[id]) 
                    {
                        continue;
                    }
                    else 
                    {
                        // Add To Sets
                        marked_constrains[id] = true;
                        current_cluster.push_back(id);
        
                        // Mark
                        current_marked[id] = true;
                        auto element = element_indices[id];
                        for (uint j = 0; j < nv; j++) 
                        {
                            for (const uint& adj_eid : vert_adj_elements[element[j]]) 
                            { 
                                current_marked[adj_eid] = true; 
                            }
                        }
                    }
                }
                
                const uint cluster_size = static_cast<uint>(current_cluster.size());
                total_marked_count += cluster_size;
        
                
                if (merge_small_cluster && cluster_size < color_threashold) 
                {
                    rest_cluster.insert(rest_cluster.end(), current_cluster.begin(), current_cluster.end());
                }
                else 
                {
                    clusterd_constraint.push_back(current_cluster);
                }
                
                if (total_marked_count == num_elements) break;
            }
        
            if (merge_small_cluster && !rest_cluster.empty()) 
            {
                clusterd_constraint.push_back(rest_cluster);
            }
        
            fast_format("Cluster Count of {} = {}", constraint_name, clusterd_constraint.size());
        };

        auto fn_get_prefix = [](SharedArray<uint>& prefix_buffer, const std::vector< std::vector<uint> >& clusterd_constraint)
        {
            const uint num_cluster = clusterd_constraint.size();
            prefix_buffer.resize(num_cluster + 1);
            uint prefix = 0;
            for (uint cluster = 0; cluster < num_cluster; cluster++)
            {
                prefix_buffer[cluster] = prefix;
                prefix += clusterd_constraint[cluster].size();
            }
            prefix_buffer[num_cluster] = prefix;
        };

        
        fn_graph_coloring_sequenced_constraint(
            mesh_data->num_edges, 
            "Distance  Spring Constraint", 
            tmp_clusterd_constraint_stretch_mass_spring, 
            mesh_data->vert_adj_edges, mesh_data->sa_edges, 2);

        fn_graph_coloring_sequenced_constraint(
            mesh_data->num_edges, 
            "Bending   Angle  Constraint", 
            tmp_clusterd_constraint_bending, 
            mesh_data->vert_adj_bending_edges, mesh_data->sa_bending_edges, 4);
            
        xpbd_data->num_clusters_stretch_mass_spring = tmp_clusterd_constraint_stretch_mass_spring.size();
        xpbd_data->num_clusters_bending = tmp_clusterd_constraint_bending.size();

        fn_get_prefix(xpbd_data->prefix_stretch_mass_spring, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(xpbd_data->prefix_bending, tmp_clusterd_constraint_bending);
        
        xpbd_data->clusterd_constraint_stretch_mass_spring.upload_2d_csr(tmp_clusterd_constraint_stretch_mass_spring);
        xpbd_data->clusterd_constraint_bending.upload_2d_csr(tmp_clusterd_constraint_bending);

    }

    // Precomputation
    {
        // Spring Constraint
        {
            xpbd_data->sa_merged_edges.resize(mesh_data->num_edges);
            xpbd_data->sa_merged_edges_rest_length.resize(mesh_data->num_edges);
            xpbd_data->sa_lambda_stretch_mass_spring.resize(mesh_data->num_edges);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_stretch_mass_spring.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_stretch_mass_spring[cluster];
                parallel_for(0, curr_cluster.size(), [&](const uint i)
                {
                    const uint constaint_idx = curr_cluster[i];
                    xpbd_data->sa_merged_edges[prefix + i] = mesh_data->sa_edges[constaint_idx];
                });
                prefix += curr_cluster.size();
            } if (prefix != mesh_data->num_edges) fast_format_err("Sum of Mass Spring Cluster Is Smaller Than Orig");
            
            // Rest spring length
            parallel_for(0, xpbd_data->sa_merged_edges.size(), [&](const uint eid)
            {
                Int2 edge = xpbd_data->sa_merged_edges[eid];
                Float3 x1 = mesh_data->sa_rest_x[edge[0]];
                Float3 x2 = mesh_data->sa_rest_x[edge[1]];
                xpbd_data->sa_merged_edges_rest_length[eid] = length_vec(x1 - x2); /// 
            });
        }

        // Bending Constraint
        {
            xpbd_data->sa_merged_bending_edges.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_merged_bending_edges_rest_angle.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_lambda_bending.resize(mesh_data->num_bending_edges);

            parallel_for(0, xpbd_data->sa_merged_bending_edges.size(), [&](const uint eid)
            {
                Int4 edge = xpbd_data->sa_merged_bending_edges[eid];
                const float eps = 1.0e-6;
    
                const uint i = edge[2];
                const uint j = edge[3];
                const uint k = edge[0];
                const uint l = edge[1];
                const Float3 x1 = mesh_data->sa_rest_x[i];
                const Float3 x2 = mesh_data->sa_rest_x[j];
                const Float3 x3 = mesh_data->sa_rest_x[k];
                const Float3 x4 = mesh_data->sa_rest_x[l];
        
                Float3 tmp;
                const float angle = Constrains::CalcGradientsAndAngle(x1, x2, x3, x4, tmp, tmp, tmp, tmp);
                if (is_nan_scalar(angle)) fast_format_err("is nan rest angle {}", eid);
    
                xpbd_data->sa_merged_bending_edges_rest_angle[eid] = angle; /// 
            });
        }
    }

}

void XpbdSolver::physics_step()
{
    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().print_xpbd_convergence ? 100 : get_scene_params().constraint_iter_count;
    const float substep_dt = get_scene_params().get_substep_dt();

    fast_format("Frame {:3} : numstep = {}, iter = {}", get_scene_params().current_frame, num_substep, constraint_iter_count);

    xpbd_data->sa_v_start = mesh_data->sa_v_frame_start;
    xpbd_data->sa_x_start = mesh_data->sa_x_frame_start;
  
    for (uint substep = 0; substep < num_substep; substep++) // 1 or 50 ?
    {   { get_scene_params().current_substep = substep; }
        
        SimClock substep_clock; substep_clock.start_clock();
        {   
            predict_position();

            collision_detection();

            {
                reset_constrains();
                reset_collision_constrains();

                for (uint iter = 0; iter < constraint_iter_count; iter++) // 200 or 1 ?
                {   
                    { get_scene_params().current_it = iter; }
                    solve_constraints();
                }
            }

            update_velocity(); 
        }
        substep_clock.end_clock();
    }

    mesh_data->sa_v_frame_end = xpbd_data->sa_v;
    mesh_data->sa_x_frame_end = xpbd_data->sa_x;
}
void XpbdSolver::reset_constrains()
{
    parallel_set(
        xpbd_data->sa_lambda_stretch_mass_spring.data(), 
        xpbd_data->sa_lambda_stretch_mass_spring.size(), 
        0.0f);
    parallel_set(
        xpbd_data->sa_lambda_bending.data(), 
        xpbd_data->sa_lambda_bending.size(), 
        0.0f);
}
void XpbdSolver::reset_collision_constrains()
{

}
void XpbdSolver::init_simulation_params()
{
    get_scene_params().print_cost_detail = true;
    get_scene_params().print_system_energy = false; // false true

    if (get_scene_params().use_substep)
    {
        get_scene_params().implicit_dt = 1.f / 60.f;
    }
    else 
    {
        get_scene_params().num_substep = 1;
        get_scene_params().constraint_iter_count = 200;
    }

    if (get_scene_params().use_small_timestep) { get_scene_params().implicit_dt = 0.001f; }
    
    get_scene_params().use_multi_buffer = false;
    get_scene_params().num_iteration = get_scene_params().num_substep * get_scene_params().constraint_iter_count;
    get_scene_params().collision_detection_frequece = 1;    

    get_scene_params().stiffness_stretch_BaraffWitkin = FEM::calcSecondLame(get_scene_params().youngs_modulus_cloth, get_scene_params().poisson_ratio_cloth); // mu;
    get_scene_params().stiffness_stretch_spring = FEM::calcSecondLame(get_scene_params().youngs_modulus_cloth, get_scene_params().poisson_ratio_cloth); // mu;
    get_scene_params().xpbd_stiffness_collision = 1e7;
    get_scene_params().balloon_scale_rate = 1.0;
    get_scene_params().stiffness_pressure = 1e6;
    
    {
        get_scene_params().stiffness_stretch_spring = 1e4;
        get_scene_params().xpbd_stiffness_collision = 1e7;
        get_scene_params().stiffness_quadratic_bending = 5e-3;
        get_scene_params().stiffness_DAB_bending = 5e-3;
    }

}
void XpbdSolver::collision_detection()
{
    // TODO
}
void XpbdSolver::predict_position()
{
    parallel_for(0, xpbd_data->sa_x.size(), [&](const uint vid)
    {
        Constrains::Core::predict_position(vid, 
            get_scene_params_array().ptr(), 
            xpbd_data->sa_x.data(), 
            xpbd_data->sa_v.data(), 
            xpbd_data->sa_x_start.data(),
            false, 
            nullptr, 
            mesh_data->sa_vert_mass.data(), 
            mesh_data->sa_is_fixed.data(), 
            get_scene_params().get_substep_dt(), 
            false);
    });
}
void XpbdSolver::update_velocity()
{
    parallel_for(0, xpbd_data->sa_x.size(), [&](const uint vid)
    {
        Constrains::Core::update_velocity(vid, 
            xpbd_data->sa_v.data(), 
            xpbd_data->sa_x.data(), 
            xpbd_data->sa_x_start.data(), 
            mesh_data->sa_x_frame_start.data(), 
            xpbd_data->sa_v.data(), 
            get_scene_params().get_substep_dt(), 
            get_scene_params().damping_cloth, 
            false);
    });
}
void XpbdSolver::solve_constraints()
{
    auto& iter_position_cloth = xpbd_data->sa_x;

    {
        for (uint i = 0; i < xpbd_data->num_clusters_stretch_mass_spring; i++) 
        {
            solve_constraint_stretch_spring(iter_position_cloth, i);
        }
    }
}
void XpbdSolver::solve_constraint_stretch_spring(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx)
{
    const uint curr_prefix = xpbd_data->prefix_stretch_mass_spring[cluster_idx];
    const uint next_prefix = xpbd_data->prefix_stretch_mass_spring[cluster_idx + 1];
    const uint num_elements_clustered = next_prefix - curr_prefix;
    
    parallel_for(0, num_elements_clustered, [&](const uint i)
    {
        const uint eid = curr_prefix + i;
        Constrains::solve_stretch_mass_spring_template(
            eid, curr_cloth_position.data(), curr_cloth_position.data(), xpbd_data->sa_x_start.data(),
            nullptr, 
            xpbd_data->sa_lambda_stretch_mass_spring.data(), mesh_data->sa_vert_mass_inv.data(), 
            xpbd_data->sa_merged_edges.data(), xpbd_data->sa_merged_edges_rest_length.data(),  // Here
            get_scene_params().stiffness_stretch_spring, get_scene_params().get_substep_dt(), false);
    }, 32);
}
void XpbdSolver::solve_constraint_bending(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx)
{

}



enum SolverType{
    SolverTypeGaussNewton,
    SolverTypeXPBD,
};

class SolverInterface
{
public:
    SolverInterface() { fast_format("Init for Solver interface"); }
    ~SolverInterface() { fast_format("Destroy Solver interface"); }  

    void set_data_pointer(BasicMeshData* mesh_ptr) 
    {
        mesh_data = mesh_ptr;
    }
    void set_solver_type(SolverType type)
    {        
        switch (type) 
        {
            case SolverTypeXPBD:
            {
                xpbd_solver.get_data_pointer(&xpbd_data, mesh_data);
                xpbd_solver.init_xpbd_system();
                xpbd_solver.init_simulation_params();
                break;
            }
            case SolverTypeGaussNewton:
            {
                break;
            }
            default:
            {
                fast_format_err("Empty solver");
                break;
            }
        }
    }

public:
    void physics_step(SolverType type);
    void restart_system();
    void save_mesh_to_obj();

private:
    BasicMeshData* mesh_data;

private:
    XpbdData xpbd_data;
    XpbdSolver xpbd_solver;
    
};

void SolverInterface::restart_system()
{
    parallel_for(0, mesh_data->num_verts, [&](uint vid)
    {
        Float3 rest_pos = mesh_data->sa_rest_x[vid];
        mesh_data->sa_x_frame_start[vid] = rest_pos;

        Float3 rest_vel = mesh_data->sa_rest_v[vid];
        mesh_data->sa_v_frame_start[vid] = rest_vel;
    });
}
void SolverInterface::physics_step(SolverType type)
{
    switch (type) 
    {
        case SolverTypeXPBD:
        {
            xpbd_solver.physics_step();
            break;
        }
        default:
        {
            fast_format_err("Emptey solver");
            break;
        }
    }

    {
        // Other operations...
    }
    mesh_data->sa_x_frame_start = mesh_data->sa_x_frame_end;
    mesh_data->sa_v_frame_start = mesh_data->sa_v_frame_end;
}
void SolverInterface::save_mesh_to_obj()
{
    const std::string filename = std::format("frame_{}.obj", get_scene_params().current_frame);

    std::string full_path = std::string(SELF_RESOURCES_PATH) + std::string("/outputs/") + filename;
    std::ofstream file(full_path, std::ios::out);

    if (file.is_open()) 
    {
        file << "# File Simulated From SIGGRAPH 2025 paper <Auto Task Scheduling for Cloth and Deformable Simulation on Heterogeneous Environments>" << std::endl;

        uint glocal_vert_id_prefix = 0;
        uint glocal_mesh_id_prefix = 0;
        
        // Cloth Part
        if (get_scene_params().draw_cloth)
        {
            // for (uint clothIdx = 0; clothIdx < cloth_data.num_cloths; clothIdx++) 
            const uint clothIdx = 0;
            {
                file << "o mesh_" << (glocal_mesh_id_prefix + clothIdx) << std::endl;
                for (uint vid = 0; vid < mesh_data->num_verts; vid++) {
                    const Float3 vertex = mesh_data->sa_x_frame_end[vid];
                    file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
                }

                for (uint fid = 0; fid < mesh_data->num_faces; fid++) {
                    const Int3 f = mesh_data->sa_faces[fid] + makeInt3(1) + makeInt3(glocal_vert_id_prefix);
                    file << "f " << f.x << " " << f.y << " " << f.z << std::endl;
                }
            }
            glocal_vert_id_prefix += mesh_data->num_verts;
            glocal_mesh_id_prefix += 1;
        }
     
        file.close();
        std::cout << "OBJ file saved: " << full_path << std::endl;
    } 
    else 
    {
        std::cerr << "Unable to open file: " << full_path << std::endl;
    }
}


int main()
{
    std::cout << "Hello, Asynchronous Iteration!" << std::endl;
    
    

    // Init metal system
    {
        create_device();

        init_command_list();

        init_scene_params();
    }
    
    // Init mesh
    BasicMeshData mesh_data;
    {
        init_mesh(&mesh_data);
    }

    // Init solver
    SolverInterface solver;
    {
        solver.set_data_pointer(&mesh_data);

        solver.set_solver_type(SolverTypeXPBD);
    }
    

    // Simulation
    {   
        solver.restart_system();

        for (uint frame = 0; frame < 20; frame++)
        {
            get_scene_params().current_frame = frame;
            
            solver.physics_step(SolverTypeXPBD);
        }

        solver.save_mesh_to_obj();
    }
    

    return 0;
}