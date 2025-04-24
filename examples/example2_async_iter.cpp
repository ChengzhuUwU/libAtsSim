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
    SharedArray<Float3> sa_x_step_start;
    SharedArray<Float3> sa_x_iter_start;


    SharedArray<Int2> sa_merged_edges; SharedArray<float> sa_merged_edges_rest_length;
    SharedArray<Int4> sa_merged_bending_edges; SharedArray<float> sa_merged_bending_edges_rest_angle;

    SharedArray<uint> clusterd_constraint_stretch_mass_spring; SharedArray<uint> prefix_stretch_mass_spring;
    SharedArray<uint> clusterd_constraint_bending; SharedArray<uint> prefix_bending; 
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

    {
        mesh_data->num_verts = num_verts; mesh_data->sa_rest_x.upload(input_mesh.model_positions); 
        mesh_data->num_faces = num_faces; mesh_data->sa_faces.upload(input_mesh.faces); 
        mesh_data->num_edges = num_edges; mesh_data->sa_edges.upload(input_mesh.edges); 
        mesh_data->num_bending_edges = num_bending_edges; mesh_data->sa_bending_edges.upload(input_mesh.bending_edges); 
    }
    {
        const float defulat_density = 0.01f;
        const float defulat_mass = defulat_density * get_scene_params().default_mass;
        mesh_data->sa_vert_mass.resize(num_verts); mesh_data->sa_vert_mass.set_data(defulat_mass);
        mesh_data->sa_vert_mass_inv.resize(num_verts); mesh_data->sa_vert_mass.set_data(1.0f / defulat_mass);
        mesh_data->sa_is_fixed.resize(num_verts); mesh_data->sa_is_fixed.set_data(0);
        mesh_data->sa_rest_v.resize(num_verts); mesh_data->sa_rest_v.set_zero();
    }
    {
        mesh_data->vert_adj_faces.resize(num_verts);
        mesh_data->vert_adj_edges.resize(num_verts);
        mesh_data->vert_adj_bending_edges.resize(num_verts);
    }

    {
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

    parallel_for(0, num_verts, [&](const uint vid)
    {
        Float3 model_position = mesh_data->sa_rest_x[vid];
        Float4x4 model_matrix = make_model_matrix(transform, rotation, scale);
        Float3 world_position = affine_position(model_matrix, model_position);
        mesh_data->sa_rest_x[vid] = world_position;
    });

}


void init_xpbd_system(BasicMeshData* mesh_data, XpbdData* xpbd_data)
{
    xpbd_data->sa_x_tilde.resize(mesh_data->num_verts); 
    xpbd_data->sa_x.resize(mesh_data->num_verts);
    xpbd_data->sa_v.resize(mesh_data->num_verts);
    xpbd_data->sa_x_step_start.resize(mesh_data->num_verts);
    xpbd_data->sa_x_iter_start.resize(mesh_data->num_verts);

    // Graph Coloring
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

        std::vector< std::vector<uint> > tmp_clusterd_constraint_stretch_mass_spring;
        std::vector< std::vector<uint> > tmp_clusterd_constraint_bending;

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

        fn_get_prefix(xpbd_data->prefix_stretch_mass_spring, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(xpbd_data->prefix_bending, tmp_clusterd_constraint_bending);

        xpbd_data->clusterd_constraint_stretch_mass_spring.upload_2d_csr(tmp_clusterd_constraint_stretch_mass_spring);
        xpbd_data->clusterd_constraint_bending.upload_2d_csr(tmp_clusterd_constraint_bending);
    }
}

class XpbdSolver
{
public:
    XpbdSolver() { fast_format("Init for XPBD Solver"); }
    ~XpbdSolver() { fast_format("Destroy XPBD Solver"); }

    // TODO: Replace to shared_ptr
    void init_solver(XpbdData* xpbd_ptr, BasicMeshData* mesh_ptr) 
    {
        xpbd_data = xpbd_ptr;
        mesh_data = mesh_ptr;
    }
    void init_simulation_params();

public:    
    void restart_system(const SharedArray<Float3>& rest_position, const SharedArray<Float3>& rest_velocity);
    void physics_step();

private:
    void collision_detection();
    void predict_position();
    void update_velocity();

private:
    void solve_constraints();
    void solve_constraint_stretch();
    void solve_constraint_bending();

private:
    XpbdData* xpbd_data;
    BasicMeshData* mesh_data;
};

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
            xpbd_data->sa_x_iter_start.data(),
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
            xpbd_data->sa_x_iter_start.data(), 
            xpbd_data->sa_x_step_start.data(), 
            xpbd_data->sa_v.data(), 
            get_scene_params().get_substep_dt(), 
            get_scene_params().damping_cloth, 
            false);
    });
}
void XpbdSolver::solve_constraints()
{

}
void XpbdSolver::solve_constraint_stretch()
{

}
void XpbdSolver::solve_constraint_bending()
{

}

int main()
{
    std::cout << "Hello, Asynchronous Iteration!" << std::endl;
    
    BasicMeshData mesh_data;
    XpbdData xpbd_data;

    // Init metal system
    {
        create_device();

        init_command_list();

        init_scene_params();
    }
    
    // Init mesh
    {
        init_mesh(&mesh_data);
    
        init_xpbd_system(&mesh_data, &xpbd_data);
    }

    // 
    
    

    return 0;
}