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
    
    SharedArray<float> sa_edges_rest_state_length;
    SharedArray<float> sa_bending_edges_rest_angle;
    SharedArray<Float4x4> sa_bending_edges_Q;

    SharedArray<uint> sa_vert_adj_verts; std::vector< std::vector<uint> > vert_adj_verts;
    SharedArray<uint> sa_vert_adj_verts_with_bending; std::vector< std::vector<uint> > vert_adj_verts_with_bending;
    SharedArray<uint> sa_vert_adj_faces; std::vector< std::vector<uint> > vert_adj_faces;
    SharedArray<uint> sa_vert_adj_edges; std::vector< std::vector<uint> > vert_adj_edges;
    SharedArray<uint> sa_vert_adj_bending_edges; std::vector< std::vector<uint> > vert_adj_bending_edges;
    
    SharedArray<float> sa_system_energy;
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

    SharedArray<Int2> sa_merged_edges; 
    SharedArray<float> sa_merged_edges_rest_length;

    SharedArray<Int4> sa_merged_bending_edges; 
    SharedArray<float> sa_merged_bending_edges_angle;
    SharedArray<Float4x4> sa_merged_bending_edges_Q;

    uint num_clusters_stretch_mass_spring = 0;
    SharedArray<uint> clusterd_constraint_stretch_mass_spring; 
    SharedArray<uint> prefix_stretch_mass_spring;
    SharedArray<float> sa_lambda_stretch_mass_spring;

    uint num_clusters_bending = 0;
    SharedArray<uint> clusterd_constraint_bending; 
    SharedArray<uint> prefix_bending; 
    SharedArray<float> sa_lambda_bending;

    // VBD
    uint num_clusters_per_vertex_bending = 0; 
    SharedArray<uint> prefix_per_vertex_bending; 
    SharedArray<uint> clusterd_per_vertex_bending; 
    SharedArray<uchar> per_vertex_bending_cluster_id; 
    SharedArray<Float4x3> sa_Hf; 

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
            sa_merged_bending_edges_rest_angle.is_empty() ||
            sa_merged_bending_edge_Q.is_empty()
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

        // Vert adj faces
        for (uint eid = 0; eid < num_faces; eid++)
        {
            auto edge = mesh_data->sa_faces[eid];
            for (uint j = 0; j < 3; j++)
                mesh_data->vert_adj_faces[edge[j]].push_back(eid);
        } 
        mesh_data->sa_vert_adj_faces.upload_2d_csr(mesh_data->vert_adj_faces); 

        // Vert adj edges
        for (uint eid = 0; eid < num_edges; eid++)
        {
            auto edge = mesh_data->sa_edges[eid];
            for (uint j = 0; j < 2; j++)
                mesh_data->vert_adj_edges[edge[j]].push_back(eid);
        } 
        mesh_data->sa_vert_adj_edges.upload_2d_csr(mesh_data->vert_adj_edges);

        // Vert adj bending-edges
        for (uint eid = 0; eid < num_bending_edges; eid++)
        {
            auto edge = mesh_data->sa_bending_edges[eid];
            for (uint j = 0; j < 4; j++)
                mesh_data->vert_adj_bending_edges[edge[j]].push_back(eid);
        }  
        mesh_data->sa_vert_adj_bending_edges.upload_2d_csr(mesh_data->vert_adj_bending_edges);

        // Vert adj verts based on 1-order connection
        mesh_data->vert_adj_verts.resize(num_verts);
        for (uint eid = 0; eid < num_edges; eid++)
        {
            auto edge = mesh_data->sa_edges[eid];
            for (uint j = 0; j < 2; j++)
            {
                const uint left = edge[j];
                const uint right = edge[1 - j];
                mesh_data->vert_adj_verts[left].push_back(right);
            }
        } 
        mesh_data->sa_vert_adj_verts.upload_2d_csr(mesh_data->vert_adj_verts);
        
        // Vert adj verts based on 1-order bending-connection
        auto insert_adj_vert = [](std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2) 
        {
            if (vid1 == vid2) std::cerr << "redudant!";
            auto& inner_list = adj_map[vid1];
            auto find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
            if (find_result == inner_list.end())
            {
                inner_list.push_back(vid2);
            }
        };
        mesh_data->vert_adj_verts_with_bending = mesh_data->vert_adj_verts;
        for (uint eid = 0; eid < mesh_data->num_bending_edges; eid++)
        {
            const Int4 edge = mesh_data->sa_bending_edges[eid];
            for (size_t i = 0; i < 4; i++) 
            {
                for (size_t j = 0; j < 4; j++)
                {
                    if (i != j) { insert_adj_vert(mesh_data->vert_adj_verts_with_bending, edge[i], edge[j]); }
                    if (i != j) { if (edge[i] == edge[j])
                    {
                        fast_format("Redundant Edge {} : {} & {}", eid, edge[i], edge[j]);
                    } }
                }
            }
        }
        mesh_data->sa_vert_adj_verts_with_bending.upload_2d_csr(mesh_data->vert_adj_verts_with_bending);
    }

    // Init energy
    {
        // Rest spring length
        mesh_data->sa_edges_rest_state_length.resize(num_edges);
        parallel_for(0, num_edges, [&](const uint eid)
        {
            Int2 edge = mesh_data->sa_edges[eid];
            Float3 x1 = mesh_data->sa_rest_x[edge[0]];
            Float3 x2 = mesh_data->sa_rest_x[edge[1]];
            mesh_data->sa_edges_rest_state_length[eid] = length_vec(x1 - x2); /// 
        });

        // Rest bending info
        mesh_data->sa_bending_edges_rest_angle.resize(num_bending_edges);
        mesh_data->sa_bending_edges_Q.resize(num_bending_edges);
        parallel_for(0, num_bending_edges, [&](const uint eid)
        {
            const Int4 edge = mesh_data->sa_bending_edges[eid];
            const Float3 vert_pos[4] = {
                mesh_data->sa_rest_x[edge[0]], 
                mesh_data->sa_rest_x[edge[1]], 
                mesh_data->sa_rest_x[edge[2]], 
                mesh_data->sa_rest_x[edge[3]]};
            
            // Rest state angle
            {
                const Float3& x1 = vert_pos[2];
                const Float3& x2 = vert_pos[3];
                const Float3& x3 = vert_pos[0];
                const Float3& x4 = vert_pos[1];
        
                Float3 tmp;
                const float angle = Constrains::CalcGradientsAndAngle(x1, x2, x3, x4, tmp, tmp, tmp, tmp);
                if (is_nan_scalar(angle)) fast_format_err("is nan rest angle {}", eid);
    
                mesh_data->sa_bending_edges_rest_angle[eid] = angle; 
            }

            // Rest state Q
            {
                auto calculateCotTheta = [](const Float3& x, const Float3& y)
                {
                    // const float scaled_cos_theta = dot_vec(x, y);
                    // const float scaled_sin_theta = (sqrt_scalar(1.0f - square_scalar(scaled_cos_theta))); 
                    const float scaled_cos_theta = dot_vec(x, y);
                    const float scaled_sin_theta = length_vec(cross_vec(x, y)); 
                    return scaled_cos_theta / scaled_sin_theta;
                };

                Float3 e0 = vert_pos[1] - vert_pos[0];
                Float3 e1 = vert_pos[2] - vert_pos[0];
                Float3 e2 = vert_pos[3] - vert_pos[0];
                Float3 e3 = vert_pos[2] - vert_pos[1];
                Float3 e4 = vert_pos[3] - vert_pos[1];
                const float cot_01 = calculateCotTheta(e0, -e1);
                const float cot_02 = calculateCotTheta(e0, -e2);
                const float cot_03 = calculateCotTheta(e0, e3);
                const float cot_04 = calculateCotTheta(e0, e4);
                const Float4 K = makeFloat4(
                    cot_03 + cot_04, 
                    cot_01 + cot_02, 
                    -cot_01 - cot_03, 
                    -cot_02 - cot_04);
                const float A_0 = 0.5f * length_vec(cross_vec(e0, e1));
                const float A_1 = 0.5f * length_vec(cross_vec(e0, e2));
                // if (is_nan_vec<Float4>(K) || is_inf_vec<Float4>(K)) fast_print_err("Q of Bending is Illigal");
                const Float4x4 m_Q = (3.f / (A_0 + A_1)) * outer_product(K, K); // Q = 3 qq^T / (A0+A1) ==> Q is symmetric
                mesh_data->sa_bending_edges_Q[eid] = m_Q; // See : A quadratic bending model for inextensible surfaces.
            }
        });
    }

    // Init vert status
    {
        mesh_data->sa_x_frame_start.resize(num_verts); mesh_data->sa_x_frame_start = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_start.resize(num_verts); mesh_data->sa_v_frame_start = mesh_data->sa_rest_v;
        mesh_data->sa_x_frame_end.resize(num_verts); mesh_data->sa_x_frame_end = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_end.resize(num_verts); mesh_data->sa_v_frame_end = mesh_data->sa_rest_v;
        mesh_data->sa_system_energy.resize(10240);
    }
    
}

class XpbdSolver
{
public:
    XpbdSolver() {}
    ~XpbdSolver() {}

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
    void compute_energy(const SharedArray<Float3>& curr_cloth_position);

private:
    void collision_detection();
    void predict_position();
    void update_velocity();
    void reset_constrains();
    void reset_collision_constrains();

private:
    SharedArray<Float4x3>& get_Hf_in_iter();
    void solve_all_constraints();
    void solve_constraint_stretch_spring(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);
    void solve_constraint_bending(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);

private:
    void solve_constraints_vbd();
    void vbd_evaluate_inertia(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_evaluate_stretch_spring(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_evaluate_bending(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_step(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_solve(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx);

private:
    XpbdData* xpbd_data;
    BasicMeshData* mesh_data;
};
static uint energy_idx = 0; 

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
            mesh_data->num_bending_edges, 
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

    // Vertex Block Descent Coloring
    {
        // Graph Coloring
        const uint num_verts_total = mesh_data->num_verts;
        xpbd_data->sa_Hf.resize(mesh_data->num_verts);

        const std::vector< std::vector<uint> >& vert_adj_verts = mesh_data->vert_adj_verts_with_bending;
        std::vector<std::vector<uint>> clusterd_vertices_bending; std::vector<uint> prefix_vertices_bending;

        auto fn_graph_coloring_pervertex = [&](const std::vector< std::vector<uint> >& vert_adj_, std::vector<std::vector<uint>>& clusterd_vertices, std::vector<uint>& prefix_vert)
        {
            std::vector<bool> marked_verts(num_verts_total, false);
            uint total_marked_count = 0;

            while (true) 
            {
                std::vector<uint> current_cluster;
                std::vector<bool> current_marked(marked_verts);

                for (uint vid = 0; vid < num_verts_total; vid++) 
                {
                    if (current_marked[vid]) 
                    {
                        continue;
                    }
                    else 
                    {
                        // Add To Sets
                        marked_verts[vid] = true;
                        current_cluster.push_back(vid);

                        // Mark
                        current_marked[vid] = true;
                        const auto& list = vert_adj_[vid];
                        for (const uint& adj_vid : list) 
                        {
                            current_marked[adj_vid] = true;
                        }
                    }
                }
                clusterd_vertices.push_back(current_cluster);
                total_marked_count += current_cluster.size();

                if (total_marked_count == num_verts_total) break;
            }
            

            prefix_vert.resize(clusterd_vertices.size() + 1); uint curr_prefix = 0;
            for (uint cluster = 0; cluster < clusterd_vertices.size(); cluster++)
            {
                prefix_vert[cluster] = curr_prefix;
                curr_prefix += clusterd_vertices[cluster].size();
            } prefix_vert[clusterd_vertices.size()] = curr_prefix;
        };

        fn_graph_coloring_pervertex(vert_adj_verts, clusterd_vertices_bending, prefix_vertices_bending);
        xpbd_data->num_clusters_per_vertex_bending = clusterd_vertices_bending.size();
        xpbd_data->prefix_per_vertex_bending.upload(prefix_vertices_bending ); 
        xpbd_data->clusterd_per_vertex_bending.upload_2d_csr(clusterd_vertices_bending);

        // Reverse map
        xpbd_data->per_vertex_bending_cluster_id.resize(mesh_data->num_verts);
        for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
        {
            const uint next_prefix = xpbd_data->clusterd_per_vertex_bending[cluster + 1];
            const uint curr_prefix = xpbd_data->clusterd_per_vertex_bending[cluster];
            const uint num_verts_cluster = next_prefix - curr_prefix;
            parallel_for(0, num_verts_cluster, [&](const uint i)
            {
                const uint vid = xpbd_data->clusterd_per_vertex_bending[curr_prefix + i];
                xpbd_data->per_vertex_bending_cluster_id[vid] = cluster;
            });
        }
        
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
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_edges[prefix + i] = mesh_data->sa_edges[eid];
                        xpbd_data->sa_merged_edges_rest_length[prefix + i] = mesh_data->sa_edges_rest_state_length[eid];
                    }
                });
                prefix += curr_cluster.size();
            } if (prefix != mesh_data->num_edges) fast_format_err("Sum of Mass Spring Cluster Is Not Equal  Than Orig");
        }

        // Bending Constraint
        {
            xpbd_data->sa_merged_bending_edges.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_merged_bending_edges_angle.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_merged_bending_edges_Q.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_lambda_bending.resize(mesh_data->num_bending_edges);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_bending.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_bending[cluster];
                parallel_for(0, curr_cluster.size(), [&](const uint i)
                {
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_bending_edges[prefix + i] = mesh_data->sa_bending_edges[eid];
                        xpbd_data->sa_merged_bending_edges_angle[prefix + i] = mesh_data->sa_bending_edges_rest_angle[eid];
                        xpbd_data->sa_merged_bending_edges_Q[prefix + i] = mesh_data->sa_bending_edges_Q[eid];
                    }
                });
                prefix += curr_cluster.size();
            } if (prefix != mesh_data->num_bending_edges) fast_format_err("Sum of Bending Cluster Is Not Equal Than Orig");
        }
    }

}
void XpbdSolver::physics_step()
{
    xpbd_data->sa_x_start = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v_start = mesh_data->sa_v_frame_start;
    xpbd_data->sa_x = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v = mesh_data->sa_v_frame_start;


    
    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().print_xpbd_convergence ? 100 : get_scene_params().constraint_iter_count;
    const float substep_dt = get_scene_params().get_substep_dt();

    mesh_data->sa_system_energy.set_zero(); energy_idx = 0;
    
    SimClock clock; clock.start_clock();

    for (uint substep = 0; substep < num_substep; substep++) // 1 or 50 ?
    {   { get_scene_params().current_substep = substep; }
        
        // SimClock substep_clock; substep_clock.start_clock();
        {   
            predict_position(); 

            collision_detection();

            // Constraint iteration part
            {
                reset_constrains();
                reset_collision_constrains();

                for (uint iter = 0; iter < constraint_iter_count; iter++) // 200 or 1 ?
                {   
                    { get_scene_params().current_it = iter; }
                    if (get_scene_params().use_xpbd_solver)     { solve_all_constraints(); }
                    else if (get_scene_params().use_vbd_solver) { solve_constraints_vbd(); }
                    else { fast_format_err("empty solver"); }
                }
            }

            update_velocity(); 
        }
        // substep_clock.end_clock();
    }
    float frame_cost = clock.end_clock();
    fast_format("Frame {:3} : cost = {:6.3f}", get_scene_params().current_frame, frame_cost);
    

    {
        if (get_scene_params().print_xpbd_convergence)
        {
            std::vector<double> list_energy(energy_idx);
            for(uint it = 0; it < list_energy.size(); it++)
            {
                list_energy[it] = mesh_data->sa_system_energy[it];
            }
            fast_print_iterator(list_energy, "Energy Convergence"); energy_idx = 0;
        }
    }




    mesh_data->sa_x_frame_end = xpbd_data->sa_x;
    mesh_data->sa_v_frame_end = xpbd_data->sa_v;
}
void XpbdSolver::compute_energy(const SharedArray<Float3>& curr_position)
{
    if (!get_scene_params().print_xpbd_convergence) return;

    double energy = 0.0;
    double energy_inertia = 0.f, energy_stretch = 0.f, energy_bending = 0.f;

    // Inertia
    {
        energy_inertia = parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
        {
            return Constrains::Energy::compute_energy_inertia(vid, 
                curr_position.data(), &get_scene_params(), mesh_data->sa_is_fixed.data(), mesh_data->sa_vert_mass.data(), 
                xpbd_data->sa_x_start.data(), xpbd_data->sa_v_start.data());
        });
    }
    
    // Stretch 
    {
        const float stiffness = get_scene_params().stiffness_stretch_spring;
        energy_stretch = parallel_for_and_reduce_sum<double>(0, mesh_data->num_edges, [&](const uint eid)
        {
            return Constrains::Energy::compute_energy_stretch_mass_spring(
                eid, curr_position.data(), 
                xpbd_data->sa_merged_edges.data(), xpbd_data->sa_merged_edges_rest_length.data(), stiffness);
        });
    }

    // Bending
    const auto bending_type = 
        (get_scene_params().use_vbd_solver // Our VBD solver only add quadratic bending implementation
        || get_scene_params().use_quadratic_bending_model) ?  
        Constrains::BendingTypeQuadratic : Constrains::BendingTypeDAB;
    const bool use_xpbd_solver = true;
    if (get_scene_params().use_bending)
    {   
        const float stiffness_bending_quadratic = get_scene_params().get_stiffness_quadratic_bending();
        const float stiffness_bending_DAB = get_scene_params().get_stiffness_DAB_bending();

        energy_bending = parallel_for_and_reduce_sum<double>(0, mesh_data->num_bending_edges, [&](const uint eid)
        {
            float energy = 0.f;
            Constrains::Energy::compute_energy_bending(bending_type, eid, curr_position.data(), 
                xpbd_data->sa_merged_bending_edges.data(), nullptr,
                nullptr, xpbd_data->sa_merged_bending_edges_Q.data(),
                xpbd_data->sa_merged_bending_edges_angle.data(), 
                stiffness_bending_DAB, stiffness_bending_quadratic, use_xpbd_solver
            );
            return energy;
        });
    }
    
    // Obstacle Collisoin
    float energy_obs_collision = 0.0f;
    // if (get_scene_params().use_obstacle_collision)
    // {
    //     const auto& obstacle_collision_data = obstacle_collision_data_cloth;
    //     const float thickness1 = 0.0f;
    //     const float thickness2 = get_scene_params().thickness_vv_obstacle;
    //     energy_obs_collision += parallel_for_and_reduce_sum<float>(0, obstacle_collision_data->collision_count[0], [&](const uint i)
    //     {
    //         return Constrains::Energy::compute_energy_collision_vf(i, curr_position.data(), obstacle_data->sa_substep_position.data(), 
    //         obstacle_collision_data->narrow_phase_list_pair_vf.data(), obstacle_collision_data->collision_count.data(), thickness2);
    //     });
    // }

    // Self Collision
    float energy_self_collision = 0.0f;
    // if (get_scene_params().use_self_collision)
    // {
    //     const auto& self_collision_data = self_collision_data_cloth;
    //     const float thickness1 = 0.0f;
    //     const float thickness2 = get_scene_params().thickness_vv_cloth;
    //     energy_self_collision = parallel_for_and_reduce_sum<float>(0, self_collision_data->collision_count[0], [&](const uint i)
    //     {
    //         return Constrains::Energy::compute_energy_collision_vv(i, curr_position.data(), 
    //         self_collision_data->narrow_phase_list_pair_vv.data(), self_collision_data->collision_count.data(), thickness2);
    //     });
    // }

    double total_energy = energy_inertia + energy_stretch + energy_bending + energy_obs_collision + energy_self_collision;

    mesh_data->sa_system_energy[energy_idx++] = total_energy;
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
    get_scene_params().print_xpbd_convergence = false; // false true

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

// XPBD constraints
void XpbdSolver::solve_constraint_stretch_spring(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx)
{
    const uint curr_prefix = xpbd_data->prefix_stretch_mass_spring[cluster_idx];
    const uint next_prefix = xpbd_data->prefix_stretch_mass_spring[cluster_idx + 1];
    const uint num_elements_clustered = next_prefix - curr_prefix;
    
    parallel_for(0, num_elements_clustered, [&](const uint i)
    {
        const uint eid = curr_prefix + i;
        Constrains::solve_stretch_mass_spring_template(
            eid, curr_cloth_position.data(), curr_cloth_position.data(), 
            xpbd_data->sa_x_start.data(),
            nullptr, 
            xpbd_data->sa_lambda_stretch_mass_spring.data(), mesh_data->sa_vert_mass_inv.data(), 
            xpbd_data->sa_merged_edges.data(), xpbd_data->sa_merged_edges_rest_length.data(),  // Here
            get_scene_params().stiffness_stretch_spring, get_scene_params().get_substep_dt(), false);
    }, 32);
}
void XpbdSolver::solve_constraint_bending(SharedArray<Float3>& curr_cloth_position, const uint cluster_idx)
{
    if (!get_scene_params().use_bending) return;

    // auto& fn_bending = Constrains::solve_bending_quadratic_template;
    auto& fn_bending = get_scene_params().use_quadratic_bending_model ? 
                                    Constrains::solve_bending_quadratic_template : 
                                    Constrains::solve_bending_DAB_template_v2;

    // fast_format("do i iter more ? substep = {} , iter = {}, cluster = {}", get_scene_params().current_substep, get_scene_params().current_it, cluster_idx);
    const float stiffness_bending_quadratic = get_scene_params().get_stiffness_quadratic_bending();
    const float stiffness_bending_DAB = get_scene_params().get_stiffness_DAB_bending();


    const uint curr_prefix = xpbd_data->prefix_bending[cluster_idx];
    const uint next_prefix = xpbd_data->prefix_bending[cluster_idx + 1];
    const uint num_elements_clustered = next_prefix - curr_prefix;

    parallel_for(0, num_elements_clustered, [&](const uint i)
    {
        const uint eid = curr_prefix + i;
        fn_bending(
                eid, curr_cloth_position.data(), curr_cloth_position.data(), 
                xpbd_data->sa_x_start.data(),
                nullptr, 
                xpbd_data->sa_lambda_bending.data(), mesh_data->sa_vert_mass_inv.data(), 
                xpbd_data->sa_merged_bending_edges.data(), nullptr, 
                xpbd_data->sa_merged_bending_edges_Q.data(), xpbd_data->sa_merged_bending_edges_angle.data(),
                stiffness_bending_quadratic, stiffness_bending_DAB, get_scene_params().get_substep_dt(), false);
    }, 32);  
}

// VBD constraints (energy)
SharedArray<Float4x3>& XpbdSolver::get_Hf_in_iter()
{
    return xpbd_data->sa_Hf;
}
void XpbdSolver::vbd_evaluate_inertia(SharedArray<Float3>& sa_iter_position, const uint cluster_idx)
{
    auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    const uint next_prefix = clusters[cluster_idx + 1];
    const uint curr_prefix = clusters[cluster_idx];
    const uint num_verts_cluster = next_prefix - curr_prefix;

    parallel_for(0, num_verts_cluster, [&](const uint i)
    {
        const uint vid = clusters[curr_prefix + i];
        Float4x3 Hf = Constrains::VBD::compute_inertia(
            vid, sa_iter_position.data(), 
            xpbd_data->sa_x_start.data(), xpbd_data->sa_v.data(), 
            mesh_data->sa_is_fixed.data(), mesh_data->sa_vert_mass.data(), &get_scene_params(),
            get_scene_params().get_substep_dt());
        get_Hf_in_iter()[vid] = Hf;
    });
}
void XpbdSolver::vbd_evaluate_stretch_spring(SharedArray<Float3>& sa_iter_position, const uint cluster_idx)
{
    auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    const uint next_prefix = clusters[cluster_idx + 1];
    const uint curr_prefix = clusters[cluster_idx];
    const uint num_verts_cluster = next_prefix - curr_prefix;
    
    parallel_for(0, num_verts_cluster, [&](const uint i)
    {
        const uint vid = clusters[curr_prefix + i];
        Float4x3 Hf = Constrains::VBD::compute_stretch_mass_spring(
                vid, sa_iter_position.data(), 
                mesh_data->sa_vert_adj_edges.data(),
                mesh_data->sa_edges.data(), mesh_data->sa_edges_rest_state_length.data(), 
                get_scene_params().stiffness_stretch_spring);
        get_Hf_in_iter()[vid] += Hf;
    }, 32);
}
void XpbdSolver::vbd_evaluate_bending(SharedArray<Float3>& sa_iter_position, const uint cluster_idx)
{
    auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    const uint next_prefix = clusters[cluster_idx + 1];
    const uint curr_prefix = clusters[cluster_idx];
    const uint num_verts_cluster = next_prefix - curr_prefix;

    parallel_for(0, num_verts_cluster, [&](const uint i)
    {
        const uint vid = clusters[curr_prefix + i];
        Float4x3 Hf = Constrains::VBD::compute_bending_quadratic(
                vid, sa_iter_position.data(),
                mesh_data->sa_vert_adj_bending_edges.data(), mesh_data->sa_bending_edges.data(), 
                mesh_data->sa_bending_edges_Q.data(), 
                get_scene_params().get_stiffness_quadratic_bending());
        get_Hf_in_iter()[vid] += Hf;
    }, 32);
}
void XpbdSolver::vbd_step(SharedArray<Float3>& sa_iter_position, const uint cluster_idx)
{
    auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    const uint next_prefix = clusters[cluster_idx + 1];
    const uint curr_prefix = clusters[cluster_idx];
    const uint num_verts_cluster = next_prefix - curr_prefix;

    parallel_for(0, num_verts_cluster, [&](const uint i)
    {
        const uint vid = clusters[curr_prefix + i];
        Float4x3 Hf = get_Hf_in_iter()[vid];
        Float3x3 H = makeFloat3x3(get(Hf, 0), get(Hf, 1), get(Hf, 2));
        Float3 f = get(Hf, 3);
        float det = determinant_mat(H);
        if (abs_scalar(det) > Epsilon)
        {
            Float3x3 H_inv = inverse_mat(H, det);
            Float3 dx = H_inv * f;
            sa_iter_position[vid] += dx;
        }
    }, 32);
}

void XpbdSolver::solve_constraints_vbd()
{
    auto& iter_position = xpbd_data->sa_x;

    if (get_scene_params().print_xpbd_convergence && get_scene_params().current_it == 0) 
    { 
        compute_energy(iter_position); 
    }

    for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
    {
        const uint next_prefix = xpbd_data->clusterd_per_vertex_bending[cluster + 1];
        const uint curr_prefix = xpbd_data->clusterd_per_vertex_bending[cluster];
        const uint num_verts_cluster = next_prefix - curr_prefix;

        vbd_evaluate_inertia(iter_position, cluster);

        vbd_evaluate_stretch_spring(iter_position, cluster);
        
        vbd_evaluate_bending(iter_position, cluster);
        
        vbd_step(iter_position, cluster);
    }

    if (get_scene_params().print_xpbd_convergence) 
    { 
        compute_energy(iter_position); 
    }
}
void XpbdSolver::solve_all_constraints()
{
    auto& iter_position_cloth = xpbd_data->sa_x;

    if (get_scene_params().print_xpbd_convergence && get_scene_params().current_it == 0) 
    { 
        compute_energy(iter_position_cloth); 
    }

    {
        for (uint i = 0; i < xpbd_data->num_clusters_stretch_mass_spring; i++) 
        {
            solve_constraint_stretch_spring(iter_position_cloth, i);
            compute_energy(iter_position_cloth); 
        }
        for (uint i = 0; i < xpbd_data->num_clusters_bending; i++) 
        {
            solve_constraint_bending(iter_position_cloth, i);
        }
    }

    if (get_scene_params().print_xpbd_convergence) 
    { 
        compute_energy(iter_position_cloth); 
    }
}

enum SolverType
{
    SolverTypeGaussNewton,
    SolverTypeXPBD,
};

class SolverInterface
{
public:
    SolverInterface() {}
    ~SolverInterface() {}  

    void set_data_pointer(BasicMeshData* mesh_ptr) 
    {
        mesh_data = mesh_ptr;
    }
    void register_solver_type(SolverType type)
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
    void save_mesh_to_obj(const std::string& addition_str = "");

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
        mesh_data->sa_x_frame_end[vid] = rest_pos;

        Float3 rest_vel = mesh_data->sa_rest_v[vid];
        mesh_data->sa_v_frame_start[vid] = rest_vel;
        mesh_data->sa_v_frame_end[vid] = rest_vel;
    });
}
void SolverInterface::physics_step(SolverType type)
{
    mesh_data->sa_x_frame_start = mesh_data->sa_x_frame_end;
    mesh_data->sa_v_frame_start = mesh_data->sa_v_frame_end;

    switch (type) 
    {
        case SolverTypeXPBD:
        {
            xpbd_solver.physics_step(); /////////////
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
    
}
void SolverInterface::save_mesh_to_obj(const std::string& addition_str)
{
    const std::string filename = std::format("frame_{}{}.obj", get_scene_params().current_frame, addition_str);

    std::string full_path = std::string(SELF_RESOURCES_PATH) + std::string("/outputs/") + filename;
    std::ofstream file(full_path, std::ios::out);

    if (file.is_open()) 
    {
        file << "# File Simulated From SIGGRAPH 2025 paper <Auto Task Scheduling for Cloth and Deformable Simulation on Heterogeneous Environments>" << std::endl;

        uint glocal_vert_id_prefix = 0;
        uint glocal_mesh_id_prefix = 0;
        
        // Cloth Part
        // if (get_scene_params().draw_cloth)
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

        solver.register_solver_type(SolverTypeXPBD);
    }



    // Some params
    {
        get_scene_params().use_substep = false;
        get_scene_params().num_substep = 1;
        get_scene_params().constraint_iter_count = 100;
        get_scene_params().use_bending = true;
        get_scene_params().use_quadratic_bending_model = true;
    }
    
    // Simulation
    solver.restart_system();
    solver.save_mesh_to_obj("");
    {
        get_scene_params().use_bending = false;
        get_scene_params().print_xpbd_convergence = true;
        get_scene_params().use_xpbd_solver = false;
        get_scene_params().use_vbd_solver = true;
    }
    {   
        for (uint frame = 0; frame < 20; frame++)
        {   get_scene_params().current_frame = frame;    
            solver.physics_step(SolverTypeXPBD);
        }
        solver.save_mesh_to_obj("_large_bending");        
    }

    // solver.restart_system();
    // {
    //     get_scene_params().use_bending = false;
    // }
    // {
    //     for (uint frame = 0; frame < 20; frame++)
    //     {   get_scene_params().current_frame = frame;    
    //         solver.physics_step(SolverTypeXPBD);
    //     }
    //     solver.save_mesh_to_obj("_no_bending");
    // }
    

    return 0;
}