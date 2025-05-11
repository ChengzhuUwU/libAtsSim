#include <iostream>
#include <tbb/tbb.h>
#include "launcher.h"
#include "shared_array.h"
#include "command_list.h"
#include "mesh_reader.h"
#include "scene_params.h"
#include "fem_energy.h"
#include "xpbd_constraints.h"

template<typename T>
using Buffer = std::vector<T>;
// using Buffer = SharedArray<T>;

struct BasicMeshData
{
    uint num_verts;
    uint num_faces;
    uint num_edges;
    uint num_bending_edges;

    Buffer<Float3> sa_rest_x;
    Buffer<Float3> sa_rest_v;

    Buffer<Float3> sa_x_frame_start;
    Buffer<Float3> sa_v_frame_start;
    Buffer<Float3> sa_x_frame_end;
    Buffer<Float3> sa_v_frame_end;

    Buffer<Int3> sa_faces;
    Buffer<Int2> sa_edges;
    Buffer<Int4> sa_bending_edges;

    Buffer<float> sa_vert_mass;
    Buffer<float> sa_vert_mass_inv;
    Buffer<uchar> sa_is_fixed;

    Buffer<float> sa_edges_rest_state_length;
    Buffer<float> sa_bending_edges_rest_angle;
    Buffer<Float4x4> sa_bending_edges_Q;

    Buffer<uint> sa_vert_adj_verts; std::vector< std::vector<uint> > vert_adj_verts;
    Buffer<uint> sa_vert_adj_verts_with_bending; std::vector< std::vector<uint> > vert_adj_verts_with_bending;
    Buffer<uint> sa_vert_adj_faces; std::vector< std::vector<uint> > vert_adj_faces;
    Buffer<uint> sa_vert_adj_edges; std::vector< std::vector<uint> > vert_adj_edges;
    Buffer<uint> sa_vert_adj_bending_edges; std::vector< std::vector<uint> > vert_adj_bending_edges;

    Buffer<float> sa_system_energy;
};

struct XpbdData
{
    Buffer<Float3> sa_x_tilde;
    Buffer<Float3> sa_x;
    Buffer<Float3> sa_v;
    Buffer<Float3> sa_v_start;
    Buffer<Float3> sa_x_start; // For calculating velocity

    Buffer<Int2> sa_merged_edges; 
    Buffer<float> sa_merged_edges_rest_length;

    Buffer<Int4> sa_merged_bending_edges; 
    Buffer<float> sa_merged_bending_edges_angle;
    Buffer<Float4x4> sa_merged_bending_edges_Q;

    uint num_clusters_stretch_mass_spring = 0;
    Buffer<uint> clusterd_constraint_stretch_mass_spring; 
    Buffer<uint> prefix_stretch_mass_spring;
    Buffer<float> sa_lambda_stretch_mass_spring;

    uint num_clusters_bending = 0;
    Buffer<uint> clusterd_constraint_bending; 
    Buffer<uint> prefix_bending; 
    Buffer<float> sa_lambda_bending;

    // VBD
    uint num_clusters_per_vertex_bending = 0; 
    Buffer<uint> prefix_per_vertex_bending; 
    Buffer<uint> clusterd_per_vertex_bending; 
    Buffer<uchar> per_vertex_bending_cluster_id; 
    Buffer<Float4x3> sa_Hf; 

    // Async
    Buffer<Float3> sa_async_iter_positions_cloth[32];
    Buffer<Float3> sa_async_begin_positions_cloth[32];
};

template<typename T>
inline void upload_from(std::vector<T>& dest, const std::vector<T>& input_data) { 
    dest.resize(input_data.size());
    std::memcpy(dest.data(), input_data.data(), dest.size() * sizeof(T));  
}
template<typename T>
inline void upload_from(SharedArray<T>& dest, const std::vector<T>& input_data) { 
    dest.upload(input_data);
}
inline uint upload_2d_csr_from(std::vector<uint>& dest, const std::vector<std::vector<uint>>& input_map) {
    uint num_outer = input_map.size();
    uint current_prefix = num_outer + 1;
    
    std::vector<uint> prefix_list(num_outer + 1);

    uint max_count = 0;
    for (uint i = 0; i < num_outer; i++) {
        const auto& inner_list = input_map[i];
        uint num_inner = inner_list.size(); max_count = std::max(max_count, num_inner);
        prefix_list[i] = current_prefix;
        current_prefix += num_inner;
    }
    uint num_data = current_prefix;
    prefix_list[num_outer] = current_prefix;
    
    dest.resize(num_data);
    std::memcpy(dest.data(), prefix_list.data(), (num_outer + 1) * sizeof(uint));

    for (uint i = 0; i < num_outer; i++) {
        const auto& inner_list = input_map[i];
        uint current_prefix = prefix_list[i];
        uint current_end = prefix_list[i + 1];
        for (uint j = current_prefix; j < current_end; j++) {
            dest[j] = inner_list[j - current_prefix];
        }
    }
    return max_count;
}
inline uint upload_2d_csr_from(SharedArray<uint>& dest, const std::vector<std::vector<uint>>& input_map) {
    return dest.upload_2d_csr(input_map);
}


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

    fast_format("Cloth : (numVerts : {}) (numFaces : {})  (numEdges : {}) (numBendingEdges : {})", 
        num_verts, num_faces, num_edges, num_bending_edges);

    // Constant scalar
    {
        mesh_data->num_verts = num_verts; upload_from(mesh_data->sa_rest_x, input_mesh.model_positions); 
        mesh_data->num_faces = num_faces; upload_from(mesh_data->sa_faces, input_mesh.faces); 
        mesh_data->num_edges = num_edges; upload_from(mesh_data->sa_edges, input_mesh.edges); 
        mesh_data->num_bending_edges = num_bending_edges; upload_from(mesh_data->sa_bending_edges, input_mesh.bending_edges); 
    }
    
    // Init vert info
    {
        // Set rest position & velocity
        {
            mesh_data->sa_rest_v.resize(num_verts); 
            parallel_for(0, num_verts, [&](const uint vid)
            {
                Float3 model_position = mesh_data->sa_rest_x[vid];
                Float4x4 model_matrix = make_model_matrix(transform, rotation, scale);
                Float3 world_position = affine_position(model_matrix, model_position);
                mesh_data->sa_rest_x[vid] = world_position;
                mesh_data->sa_rest_v[vid] = Zero3;
            });
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
                mesh_data->sa_vert_mass[vid] = (defulat_mass);
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
        upload_2d_csr_from(mesh_data->sa_vert_adj_faces, mesh_data->vert_adj_faces); 

        // Vert adj edges
        for (uint eid = 0; eid < num_edges; eid++)
        {
            auto edge = mesh_data->sa_edges[eid];
            for (uint j = 0; j < 2; j++)
                mesh_data->vert_adj_edges[edge[j]].push_back(eid);
        } 
        upload_2d_csr_from(mesh_data->sa_vert_adj_edges, mesh_data->vert_adj_edges);

        // Vert adj bending-edges
        for (uint eid = 0; eid < num_bending_edges; eid++)
        {
            auto edge = mesh_data->sa_bending_edges[eid];
            for (uint j = 0; j < 4; j++)
                mesh_data->vert_adj_bending_edges[edge[j]].push_back(eid);
        }  
        upload_2d_csr_from(mesh_data->sa_vert_adj_bending_edges, mesh_data->vert_adj_bending_edges);

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
        upload_2d_csr_from(mesh_data->sa_vert_adj_verts, mesh_data->vert_adj_verts);
        
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
        upload_2d_csr_from(mesh_data->sa_vert_adj_verts_with_bending, mesh_data->vert_adj_verts_with_bending);
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

class CpuSolver
{
public:
    CpuSolver() {}
    ~CpuSolver() {}

    // TODO: Replace to shared_ptr
    void get_data_pointer(XpbdData* xpbd_ptr, BasicMeshData* mesh_ptr) 
    {
        xpbd_data = xpbd_ptr;
        mesh_data = mesh_ptr;
    }
    void init_xpbd_system();
    static void init_simulation_params();

public:    
    void physics_step_vbd();
    void physics_step_xpbd();
    void physics_step_vbd_async();
    void fn_dispatch(const Launcher::LaunchParam& param);
    void compute_energy(const Buffer<Float3>& curr_cloth_position);

private:
    void collision_detection();
    void predict_position();
    void update_velocity();
    void reset_constrains();
    void reset_collision_constrains();

private:
    Buffer<Float4x3>& get_Hf();
    void solve_constraints_XPBD();
    void solve_constraint_stretch_spring(Buffer<Float3>& curr_cloth_position, const uint cluster_idx);
    void solve_constraint_bending(Buffer<Float3>& curr_cloth_position, const uint cluster_idx);

private:
    void solve_constraints_VBD();
    void vbd_evaluate_inertia(Buffer<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_evaluate_stretch_spring(Buffer<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_evaluate_bending(Buffer<Float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_step(Buffer<Float3>& curr_cloth_position, const uint cluster_idx);

private:
    XpbdData* xpbd_data;
    BasicMeshData* mesh_data;
};
static uint energy_idx = 0; 

void CpuSolver::init_xpbd_system()
{
    xpbd_data->sa_x_tilde.resize(mesh_data->num_verts); 
    xpbd_data->sa_x.resize(mesh_data->num_verts);
    xpbd_data->sa_v.resize(mesh_data->num_verts); xpbd_data->sa_v = mesh_data->sa_rest_v;
    xpbd_data->sa_v_start.resize(mesh_data->num_verts); xpbd_data->sa_v_start = mesh_data->sa_rest_v;
    xpbd_data->sa_x_start.resize(mesh_data->num_verts);

    for (auto& buffer : xpbd_data->sa_async_iter_positions_cloth)  buffer.resize(mesh_data->num_verts);
    for (auto& buffer : xpbd_data->sa_async_begin_positions_cloth) buffer.resize(mesh_data->num_verts);

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

        auto fn_get_prefix = [](auto& prefix_buffer, const std::vector< std::vector<uint> >& clusterd_constraint)
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
        
        upload_2d_csr_from(xpbd_data->clusterd_constraint_stretch_mass_spring, tmp_clusterd_constraint_stretch_mass_spring);
        upload_2d_csr_from(xpbd_data->clusterd_constraint_bending, tmp_clusterd_constraint_bending);

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
        upload_from(xpbd_data->prefix_per_vertex_bending, prefix_vertices_bending ); 
        upload_2d_csr_from(xpbd_data->clusterd_per_vertex_bending, clusterd_vertices_bending);

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
void CpuSolver::reset_constrains()
{
    auto fn_reset_template = [&](Buffer<float>& buffer)
    {
        parallel_set(
            buffer.data(), 
            buffer.size(), 
            0.0f);
    };

    fn_reset_template(xpbd_data->sa_lambda_stretch_mass_spring);
    fn_reset_template(xpbd_data->sa_lambda_bending);
}
void CpuSolver::reset_collision_constrains()
{

}
void CpuSolver::init_simulation_params()
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
void CpuSolver::collision_detection()
{
    // TODO
}
void CpuSolver::predict_position()
{
    parallel_for(0, mesh_data->num_verts, [&](const uint vid)
    {
        Constrains::Core::predict_position(vid, 
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
void CpuSolver::update_velocity()
{
    parallel_for(0, mesh_data->num_verts, [&](const uint vid)
    {
        Constrains::Core::update_velocity(vid, 
            xpbd_data->sa_v.data(), 
            xpbd_data->sa_x.data(), 
            xpbd_data->sa_x_start.data(), 
            mesh_data->sa_x_frame_start.data(), 
            xpbd_data->sa_v_start.data(), 
            get_scene_params().get_substep_dt(), 
            get_scene_params().damping_cloth, 
            false);
    });
}
void CpuSolver::compute_energy(const Buffer<Float3>& curr_position)
{
    if (!get_scene_params().print_xpbd_convergence) return;
    // fast_format("buffer size = {}", curr_position.size());

    double energy = 0.0;
    double energy_inertia = 0.f, energy_stretch = 0.f, energy_bending = 0.f;

    // Inertia
    {
        energy_inertia = parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
        {
            return Constrains::Energy::compute_energy_inertia(vid, 
                curr_position.data(), 
                &get_scene_params(), 
                mesh_data->sa_is_fixed.data(), 
                mesh_data->sa_vert_mass.data(), 
                xpbd_data->sa_x_start.data(), 
                xpbd_data->sa_v_start.data());
        });
    }
    
    // Stretch 
    {
        const float stiffness = get_scene_params().stiffness_stretch_spring;
        energy_stretch = parallel_for_and_reduce_sum<double>(0, mesh_data->num_edges, [&](const uint eid)
        {
            return Constrains::Energy::compute_energy_stretch_mass_spring(
                eid, curr_position.data(), 
                xpbd_data->sa_merged_edges.data(), 
                xpbd_data->sa_merged_edges_rest_length.data(), 
                stiffness);
        });
    }

    // Bending
    if (get_scene_params().use_bending)
    {   
        const auto bending_type = 
            (get_scene_params().use_vbd_solver // Our VBD solver only add quadratic bending implementation
            || get_scene_params().use_quadratic_bending_model) ?  
            Constrains::BendingTypeQuadratic : Constrains::BendingTypeDAB;
        const bool use_xpbd_solver = get_scene_params().use_xpbd_solver;

        const float stiffness_bending_quadratic = get_scene_params().get_stiffness_quadratic_bending();
        const float stiffness_bending_DAB = get_scene_params().get_stiffness_DAB_bending();

        energy_bending = parallel_for_and_reduce_sum<double>(0, mesh_data->num_bending_edges, [&](const uint eid)
        {
            float energy = 0.f;
            Constrains::Energy::compute_energy_bending(bending_type, eid, curr_position.data(), 
                xpbd_data->sa_merged_bending_edges.data(), 
                nullptr,
                nullptr, 
                xpbd_data->sa_merged_bending_edges_Q.data(),
                xpbd_data->sa_merged_bending_edges_angle.data(), 
                stiffness_bending_DAB, 
                stiffness_bending_quadratic, 
                use_xpbd_solver
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


// XPBD constraints
void CpuSolver::solve_constraint_stretch_spring(Buffer<Float3>& curr_cloth_position, const uint cluster_idx)
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
void CpuSolver::solve_constraint_bending(Buffer<Float3>& curr_cloth_position, const uint cluster_idx)
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
Buffer<Float4x3>& CpuSolver::get_Hf()
{
    return xpbd_data->sa_Hf;
}
void CpuSolver::vbd_evaluate_inertia(Buffer<Float3>& sa_iter_position, const uint cluster_idx)
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
        get_Hf()[vid] = Hf;
    });
}
void CpuSolver::vbd_evaluate_stretch_spring(Buffer<Float3>& sa_iter_position, const uint cluster_idx)
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
        get_Hf()[vid] += Hf;
    }, 32);
}
void CpuSolver::vbd_evaluate_bending(Buffer<Float3>& sa_iter_position, const uint cluster_idx)
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
        get_Hf()[vid] += Hf;
    }, 32);
}
void CpuSolver::vbd_step(Buffer<Float3>& sa_iter_position, const uint cluster_idx)
{
    auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    const uint next_prefix = clusters[cluster_idx + 1];
    const uint curr_prefix = clusters[cluster_idx];
    const uint num_verts_cluster = next_prefix - curr_prefix;

    parallel_for(0, num_verts_cluster, [&](const uint i)
    {
        const uint vid = clusters[curr_prefix + i];
        Float4x3 Hf = get_Hf()[vid];
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


void CpuSolver::physics_step_xpbd()
{
    xpbd_data->sa_x_start = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v_start = mesh_data->sa_v_frame_start;
    xpbd_data->sa_x = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v = mesh_data->sa_v_frame_start;


    
    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;

    std::memset(mesh_data->sa_system_energy.data(), 0, mesh_data->sa_system_energy.size() * sizeof(float));
    energy_idx = 0;
    
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
                    if (get_scene_params().use_xpbd_solver)     { solve_constraints_XPBD(); }
                    else { fast_format_err("empty solver"); }
                }
            }

            update_velocity(); 
        }
        // substep_clock.end_clock();
    }
    float frame_cost = clock.end_clock();
    // fast_format("Frame {:3} : cost = {:6.3f}", get_scene_params().current_frame, frame_cost);
    

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
void CpuSolver::physics_step_vbd()
{
    xpbd_data->sa_x_start = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v_start = mesh_data->sa_v_frame_start;
    xpbd_data->sa_x = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v = mesh_data->sa_v_frame_start;


    
    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;

    std::memset(mesh_data->sa_system_energy.data(), 0, mesh_data->sa_system_energy.size() * sizeof(float));
    energy_idx = 0;
    
    SimClock clock; clock.start_clock();

    for (uint substep = 0; substep < num_substep; substep++) // 1 or 50 ?
    {   { get_scene_params().current_substep = substep; }
        
        // SimClock substep_clock; substep_clock.start_clock();
        {   
            predict_position(); 

            collision_detection();

            // Constraint iteration part
            {
                for (uint iter = 0; iter < constraint_iter_count; iter++) // 200 or 1 ?
                {   
                    { get_scene_params().current_it = iter; }
                    if (get_scene_params().use_vbd_solver) { solve_constraints_VBD(); }
                    else { fast_format_err("empty solver"); }
                }
            }

            update_velocity(); 
        }
        // substep_clock.end_clock();
    }
    float frame_cost = clock.end_clock();
    // fast_format("Frame {:3} : cost = {:6.3f}", get_scene_params().current_frame, frame_cost);
    

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
void CpuSolver::fn_dispatch(const Launcher::LaunchParam& param)
{
    // Asynchronous iteration part
    constexpr uint max_buffer_count = 32; 
    constexpr bool print_buffer_idx = false;
    auto fn_get_iter_buffer = [&](const uint buffer_idx) -> Buffer<Float3>& 
    { 
        // if constexpr (print_buffer_idx) fast_format("Iter buffer {} ({}) size = {}", buffer_idx, buffer_idx % max_buffer_count, xpbd_data->sa_async_iter_positions_cloth[buffer_idx % max_buffer_count].size());
        return xpbd_data->sa_async_iter_positions_cloth[buffer_idx % max_buffer_count];
    };
    auto fn_get_begin_buffer = [&](const uint buffer_idx) -> Buffer<Float3>& 
    { 
        // if constexpr (print_buffer_idx) fast_format("Begin buffer {} ({}) size = {}", buffer_idx, buffer_idx % max_buffer_count, xpbd_data->sa_async_begin_positions_cloth[buffer_idx % max_buffer_count].size());
        return xpbd_data->sa_async_begin_positions_cloth[buffer_idx % max_buffer_count];
    };
    auto fn_copy_to_start_and_iter = [&](const Buffer<Float3>& input_array, const uint output_buffer_idx)
    {
        Buffer<Float3>& out1 = fn_get_begin_buffer(output_buffer_idx);
        Buffer<Float3>& out2 = fn_get_iter_buffer(output_buffer_idx);
        if constexpr (print_buffer_idx) fast_format("fn_copy_to_start_and_iter from {} to {}/{}", input_array.size(), out1.size(), out2.size());
        parallel_for(0, input_array.size(), [&](const uint vid)
        {   
            Float3 input_vec = input_array[vid];
            out1[vid] = input_vec;
            out2[vid] = input_vec;
        });
    };
    auto fn_cloth_constraint_prev_func = [&](const Launcher::LaunchParam& param)
    {
        if constexpr (print_buffer_idx) fast_format("Prev get Buffer {}", param.buffer_idx);
        const float weight = 0.5f;
        
        if (!param.input_buffer_idxs.empty() && param.left_buffer_idx != -1u) // Weight from left and input
        {
            for (const uint input_buffer_idx : param.input_buffer_idxs)
            {
                if constexpr (print_buffer_idx) fast_format("Weight : from {} and {}", input_buffer_idx, param.left_buffer_idx);
                auto& begin_buffer = param.is_allocated_to_main_device ? fn_get_begin_buffer(input_buffer_idx) : fn_get_begin_buffer(param.left_buffer_idx);
                parallel_for(0, mesh_data->num_verts, [&](const uint vid)
                {
                    Constrains::Core::read_and_solve_conflict(vid, 
                        begin_buffer.data(), 
                        begin_buffer.data(), 
                        fn_get_iter_buffer(input_buffer_idx).data(), 
                        fn_get_iter_buffer(param.left_buffer_idx).data(), 
                        weight);
                });
            }
        }
        else if (!param.input_buffer_idxs.empty()) // Copy from input
        {
            if constexpr (print_buffer_idx) fast_format("Copy left  : from {} to {}", param.input_buffer_idxs.back(), param.buffer_idx);
            fn_copy_to_start_and_iter(fn_get_iter_buffer(param.input_buffer_idxs.back()), param.buffer_idx);
        }
        else if (param.left_buffer_idx != -1u && param.left_buffer_idx != Launcher::input_buffer_mask) // Copy from left
        {
            if constexpr (print_buffer_idx) fast_format("Copy input: from {} to {}", param.left_buffer_idx, param.buffer_idx);
            fn_copy_to_start_and_iter(fn_get_iter_buffer(param.left_buffer_idx), param.buffer_idx);
        }
        else if (param.left_buffer_idx == Launcher::input_buffer_mask) 
        {
            if constexpr (print_buffer_idx) fast_format("Copy input: from sa_x to {}", param.buffer_idx);
            fn_copy_to_start_and_iter(xpbd_data->sa_x, param.buffer_idx);
        }

        if (get_scene_params().print_xpbd_convergence && param.iter_idx == 0 &&param.cluster_idx == 0)
        { 
            compute_energy(fn_get_iter_buffer(param.buffer_idx)); 
        }
    };
    auto fn_cloth_constraint_post_func = [&](const Launcher::LaunchParam& param)
    {
        if constexpr (print_buffer_idx) fast_format("Post get Buffer {}", param.buffer_idx);

        if (get_scene_params().print_xpbd_convergence) 
        {
            if (
                param.function_id == Launcher::id_xpbd_constraint_self_collision_vv_half_cloth // Last task of XPBD (collision)
                || (param.cluster_idx == xpbd_data->num_clusters_per_vertex_bending - 1 && param.function_id == Launcher::id_vbd_all_in_one)
                || param.function_id == Launcher::id_xpbd_constraint_last_node
                ) 
            {
                compute_energy(fn_get_iter_buffer(param.buffer_idx)); 
            }
        }
    };
    
    // Register Implementation

    // auto fn_launch = [&](const Launcher::LaunchParam& param) // Why cant i use it in lambda ???
    {
        switch (param.function_id) 
        {
            case Launcher::id_xpbd_predict_position : 
            { 
                predict_position() ; break; 
            }
            case Launcher::id_xpbd_update_velocity : 
            { 
                update_velocity(); break; 
            }
            case Launcher::id_xpbd_reset_constrains :  
            { 
                reset_constrains() ; break; 
            }
            case Launcher::id_xpbd_reset_collision_constrains : 
            { 
                reset_collision_constrains() ; break; 
            }
            case Launcher::id_xpbd_constraint_last_node :
            {
                fn_cloth_constraint_prev_func(param);
                { 

                }
                fn_cloth_constraint_post_func(param);
                parallel_copy(xpbd_data->sa_x.data(), fn_get_iter_buffer(param.buffer_idx).data(), xpbd_data->sa_x.size());
                break;
            }
            case Launcher::id_xpbd_copy_to_cpu_gpu:
            {
                fn_copy_to_start_and_iter(xpbd_data->sa_x, 0);
                fn_copy_to_start_and_iter(xpbd_data->sa_x, 1);
                break;
            }
            case Launcher::id_vbd_all_in_one:
            {
                auto& iter_position = fn_get_iter_buffer(param.buffer_idx); 
                
                const uint cluster = param.cluster_idx;
                
                fn_cloth_constraint_prev_func(param);
                {
                    vbd_evaluate_inertia(iter_position, cluster);

                    vbd_evaluate_stretch_spring(iter_position, cluster);
                    
                    vbd_evaluate_bending(iter_position, cluster);
                    
                    vbd_step(iter_position, cluster);
                }
                // const uint iter = param.iter_idx;
                // if (cluster == xpbd_data->num_clusters_per_vertex_bending - 1) 
                //     chebyshev_step(iter_position, iter); // chebyshev acceleration is not supported, which may be future work
                fn_cloth_constraint_post_func(param);
                break;
            }
            default:
            {
                fast_print_err("Illigal Input", Launcher::taskNames.at(param.function_id));
                break;
            }
        }
    };
}
void CpuSolver::physics_step_vbd_async()
{
    xpbd_data->sa_x_start = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v_start = mesh_data->sa_v_frame_start;
    xpbd_data->sa_x = mesh_data->sa_x_frame_start;
    xpbd_data->sa_v = mesh_data->sa_v_frame_start;

    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;

    std::memset(mesh_data->sa_system_energy.data(), 0, mesh_data->sa_system_energy.size() * sizeof(float)); energy_idx = 0;

    Launcher::Scheduler scheduler;

    // Register DAG and implementation
    {
        Launcher::Implementation ipm_xpbd_cpu(Launcher::DeviceTypeCpu, [&](const Launcher::LaunchParam& param) { fn_dispatch(param); });
        Launcher::Implementation imp_xpbd_gpu(Launcher::DeviceTypeGpu, [&](const Launcher::LaunchParam& param) { fn_dispatch(param); }); // Actually is CPU-implementation, you can replace it to your GPU-implementation's interface

        //
        // Register DAG
        //
        {
            std::vector<Launcher::Implementation> implementation_list_xpbd_cpu_and_gpu = {ipm_xpbd_cpu, imp_xpbd_gpu};
            
            uint tid_xpbd_predict_position = scheduler.add_task(Launcher::Task(Launcher::id_xpbd_predict_position, 0, implementation_list_xpbd_cpu_and_gpu));
        
            std::vector<uint> constraint_task_orders;
            std::vector< std::vector<uint> > constraint_tasks;
    
            auto fn_connect_single_single = [&](const uint left, const uint right) { scheduler.set_connect(left, right); };
            auto fn_connect_single_multiple = [&](const uint left, const std::vector<uint>& rights) { for (const uint& right : rights) scheduler.set_connect(left, right); };
            auto fn_connect_multiple_single = [&](const std::vector<uint>& lefts, const uint right) { for (const uint& left : lefts) scheduler.set_connect(left, right); };
            auto fn_connect_multiple_multiple = [&](const std::vector<uint>& lefts, const std::vector<uint>& rights) 
            { 
                for (const uint& left : lefts) 
                {
                    for (const uint& right : rights)
                    {
                        scheduler.set_connect(left, right); 
                    }
                }
            };
    
            for (uint iter = 0; iter < constraint_iter_count; iter++)
            {
                std::vector<uint> curr_tasks;
                for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
                {
                    uint tid_vbd_vbd_all_in_one = scheduler.add_task(
                        Launcher::Task(Launcher::id_vbd_all_in_one, iter, -1u, cluster, implementation_list_xpbd_cpu_and_gpu));
            
                    fn_connect_single_single(tid_xpbd_predict_position, tid_vbd_vbd_all_in_one);
                
                    curr_tasks.push_back(tid_vbd_vbd_all_in_one);
                }
                constraint_tasks.push_back(curr_tasks);
                constraint_task_orders.insert(constraint_task_orders.end(), curr_tasks.begin(), curr_tasks.end());
                
                if (iter >= 1)
                {
                    const auto& prev_tasks = constraint_tasks[iter - 1];
                    for (uint cluster = 0; cluster < prev_tasks.size(); cluster++)
                    {
                        // Single cluster should maintain sequency
                        fn_connect_single_single(prev_tasks[cluster], curr_tasks[cluster]);   
                    }
                }
            }
    
            scheduler.set_constraint_task_orders(constraint_task_orders);
    
            // After All Iteration => Assemble And Update Velocity
            {
                uint last_node = scheduler.add_task(Launcher::Task(Launcher::id_xpbd_constraint_last_node, 0, implementation_list_xpbd_cpu_and_gpu));
                uint tid_xpbd_update_velocity = scheduler.add_task(Launcher::Task(Launcher::id_xpbd_update_velocity, 0, implementation_list_xpbd_cpu_and_gpu));
                fn_connect_multiple_single(constraint_tasks.back(), last_node);
                scheduler.set_connect(last_node, tid_xpbd_update_velocity);
            }
        }
    }

    // Set computation matrix
    {
        std::vector< std::pair<Launcher::FunctionID, uint> > list_task_id = { };
        std::vector< std::vector<double> > list_cost; 
        std::vector< double > cost_total;

        list_task_id = {
            { Launcher::id_xpbd_predict_position, 0 }, 
            { Launcher::id_xpbd_update_velocity, 0 }, 
            { Launcher::id_xpbd_copy_to_cpu_gpu, 0 }, 
            { Launcher::id_xpbd_constraint_last_node, 0 }, 
            { Launcher::id_vbd_all_in_one, 0 },
        };
        for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
        {
            list_task_id.push_back({Launcher::id_vbd_all_in_one, cluster});
        }
        for (uint i = 0; i < list_task_id.size(); i++)
        {
            list_cost.push_back({0.1f, 0.1f});
        }
        scheduler.profile_from(list_task_id, list_cost, cost_total);
    }

    // Set communication matrix
    {
        scheduler.communication_cost_matrix_uma = {
            {0.0f, 0.1f},
            {0.1f, 0.0f},
        };
        scheduler.communication_speed_matrix = {};
        scheduler.communication_startup = {0, 0}; // First call cost
    }

    // Make scheduling
    if (scheduler.topological_sort()) 
    {
        // scheduler.print_sort_by_typology();

        scheduler.standardizing_dag();

        scheduler.scheduler_dag();
        
        if (get_scene_params().current_frame == 0 && get_scene_params().constraint_iter_count < 20) scheduler.print_schedule_to_graph_xpbd();
        if (get_scene_params().current_frame == 0) scheduler.print_speedups_to_each_device();

        scheduler.make_wait_events();
    }
    
    
    auto fn_task_to_param = [](const Launcher::Task& task) 
    { 
        // task.print_with_cluster(0);
        return Launcher::LaunchParam{
            .function_id = task.func_id, 
            .iter_idx = task.iter_idx, 
            .cluster_idx = task.cluster_idx, 
            .is_allocated_to_main_device = task.is_allocated_to_main_device,
            .buffer_idx = task.buffer_idx, 
            .left_buffer_idx = task.buffer_left, 
            .input_buffer_idxs = task.buffer_ins, 
        }; 
    };
    
    SimClock clock; clock.start_clock();
    for (uint substep = 0; substep < num_substep; substep++) // 1 or 50 ?
    {   { get_scene_params().current_substep = substep; }
        
        // SimClock substep_clock; substep_clock.start_clock();
        {   
            scheduler.launch(Launcher::Scheduler::LaunchModeSequeceHetero, fn_task_to_param, false);
        }
        // substep_clock.end_clock();
    }
    float frame_cost = clock.end_clock();
    // fast_format("Frame {:3} : cost = {:6.3f}", get_scene_params().current_frame, frame_cost);
    

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
void CpuSolver::solve_constraints_VBD()
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
void CpuSolver::solve_constraints_XPBD()
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
    SolverTypeXPBD_CPU,
    SolverTypeVBD_CPU,
    SolverTypeVBD_async,
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
        if (type == SolverTypeGaussNewton)
        {
            fast_format_err("Empty NewtonSolver implementation");
        }
        else 
        {
            cpu_solver.get_data_pointer(&xpbd_data, mesh_data);
            cpu_solver.init_xpbd_system();

            CpuSolver::init_simulation_params();
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
    CpuSolver cpu_solver;
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
        case SolverTypeXPBD_CPU:
        {
            cpu_solver.physics_step_xpbd(); /////////////
            break;
        }
        case SolverTypeVBD_CPU:
        {
            cpu_solver.physics_step_vbd(); /////////////
            break;
        }
        case SolverTypeVBD_async:
        {
            cpu_solver.physics_step_vbd_async(); /////////////
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

    std::string full_directory = std::string(SELF_RESOURCES_PATH) + std::string("/outputs/");
    
    {
        std::filesystem::path dir_path(full_directory);
        if (!std::filesystem::exists(dir_path)) 
        {
            try 
            {
                std::filesystem::create_directories(dir_path);
                std::cout << "Created directory: " << dir_path << std::endl;
            } 
            catch (const std::filesystem::filesystem_error& e) 
            {
                std::cerr << "Error creating directory: " << e.what() << std::endl;
                return;
            }
        }
    }

    std::string full_path = full_directory + filename;
    std::ofstream file(full_path, std::ios::out);

    if (file.is_open()) 
    {
        file << "# File Simulated From SIGGRAPH 2025 paper <Automatic Task Scheduling for Cloth and Deformable Simulation on Heterogeneous Environments>" << std::endl;

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

        solver.register_solver_type(SolverTypeVBD_CPU);
    }

    // Some params
    {
        get_scene_params().use_substep = false;
        get_scene_params().num_substep = 1;
        get_scene_params().constraint_iter_count = 100; // 
        get_scene_params().use_bending = true;
        get_scene_params().use_quadratic_bending_model = true;
        get_scene_params().print_xpbd_convergence = true;
        get_scene_params().use_xpbd_solver = false;
        get_scene_params().use_vbd_solver = true;
    }


    
    // Synchronous Implementation
    {
        solver.restart_system();
        solver.save_mesh_to_obj(""); 
        fast_format("");
        fast_format("");
        fast_format("Sync part");
    }
    {   
        for (uint frame = 0; frame < 10; frame++)
        {   get_scene_params().current_frame = frame; fast_format("     Sync frame {}", frame);   

            if (frame != 9) get_scene_params().print_xpbd_convergence = false;
            if (frame == 9) get_scene_params().print_xpbd_convergence = true; // Print the energy convergence in frame 10
            solver.physics_step(SolverTypeVBD_CPU);
        }
    }
    {
        solver.save_mesh_to_obj("_sync"); 
    }       


    // Asynchronous Implementation
    {
        solver.restart_system();
        fast_format("Async part");
    }
    {
        for (uint frame = 0; frame < 10; frame++)
        {   get_scene_params().current_frame = frame; fast_format("    Async frame {}", frame);
            
            if (frame != 9) get_scene_params().print_xpbd_convergence = false;
            if (frame == 9) get_scene_params().print_xpbd_convergence = true;
            solver.physics_step(SolverTypeVBD_async);
        }
    }
    {
        solver.save_mesh_to_obj("_async");
    }
    

    return 0;
}