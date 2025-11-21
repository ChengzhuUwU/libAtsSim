#include "command_list.h"
#include "fem_energy.h"
#include "graph_coloring_cpu.h"
#include "graph_coloring_gpu.h"
#include "launcher.h"
#include "lbvh_interface.h"
#include "mesh_reader.h"
#include "obstacle_data.h"
#include "scene_params.h"
#include "shared/vivace_kernel.h"
#include "shared_array.h"
#include "sim_data.h"
#include "struct_to_string.h"
#include "xpbd_constraints.h"
#include "xpbd_data.h"
#include <iostream>
#include <tbb/tbb.h>

template<typename T>
using Buffer = SharedArray<T>;
// using Buffer = std::vector<T>;

struct InputTriangleMesh {
    std::string mesh_name;
    std::string mtl_filename = "";
    std::string use_mtl_name = "";
    TriangleMeshData mesh;
    std::vector<bool> fixed_points;
    uint m_inner_order;
    Float3 m_translation;
    Float3 m_rotation;
    Float3 m_scale;
    Float4x4 m_matrix;// MVP中的M
    InputTriangleMesh() {}
    InputTriangleMesh(
        const std::string &mesh_name,
        // const std::string& mtl_filename,
        // const std::string& use_mtl_name,
        const TriangleMeshData &mesh,
        const std::vector<bool> &fixed_points,
        const Float3 &m_translation,
        const Float3 &m_rotation,
        const Float3 &m_scale) : mesh_name(mesh_name),
                                 // mtl_filename(mtl_filename),
                                 // use_mtl_name(use_mtl_name),
                                 mesh(mesh),
                                 fixed_points(fixed_points),
                                 m_inner_order(0),
                                 m_translation(m_translation),
                                 m_rotation(m_rotation),
                                 m_scale(m_scale),
                                 m_matrix(make_model_matrix(m_translation, m_rotation, m_scale)) {}
};
struct InputTetrahedralMesh {
    std::vector<Float3> positions;
    std::vector<bool> fixed_points;
    std::vector<Int4> tets;
    std::vector<uint> inner_tets;
    std::vector<uint> outer_tets;
    std::vector<Int3> surface_faces;
    std::vector<Int2> surface_edges;
    std::vector<uint> surface_verts;
    std::string mesh_name;
    std::string mtl_filename;
    std::string use_mtl_name;

    Float3 m_translation;
    Float3 m_rotation;
    Float3 m_scale;
    Float4x4 m_matrix;

    void release() {
        positions.shrink_to_fit();
        tets.shrink_to_fit();
        inner_tets.shrink_to_fit();
        outer_tets.shrink_to_fit();
        surface_faces.shrink_to_fit();
        surface_edges.shrink_to_fit();
        surface_verts.shrink_to_fit();
        mesh_name.shrink_to_fit();
    }
};

template<typename T>
inline void upload_from(std::vector<T> &dest,
                        const std::vector<T> &input_data) {
    dest.resize(input_data.size());
    std::memcpy(dest.data(), input_data.data(), dest.size() * sizeof(T));
}
template<typename T>
inline void upload_from(SharedArray<T> &dest,
                        const std::vector<T> &input_data) {
    dest.upload(input_data);
}
inline uint
upload_2d_csr_from(std::vector<uint> &dest,
                   const std::vector<std::vector<uint>> &input_map) {
    uint num_outer = input_map.size();
    uint current_prefix = num_outer + 1;

    std::vector<uint> prefix_list(num_outer + 1);

    uint max_count = 0;
    for (uint i = 0; i < num_outer; i++) {
        const auto &inner_list = input_map[i];
        uint num_inner = inner_list.size();
        max_count = std::max(max_count, num_inner);
        prefix_list[i] = current_prefix;
        current_prefix += num_inner;
    }
    uint num_data = current_prefix;
    prefix_list[num_outer] = current_prefix;

    dest.resize(num_data);
    std::memcpy(dest.data(), prefix_list.data(), (num_outer + 1) * sizeof(uint));

    for (uint i = 0; i < num_outer; i++) {
        const auto &inner_list = input_map[i];
        uint current_prefix = prefix_list[i];
        uint current_end = prefix_list[i + 1];
        for (uint j = current_prefix; j < current_end; j++) {
            dest[j] = inner_list[j - current_prefix];
        }
    }
    return max_count;
}
inline uint
upload_2d_csr_from(SharedArray<uint> &dest,
                   const std::vector<std::vector<uint>> &input_map) {
    return dest.upload_2d_csr(input_map);
}

void preprocess_tet_mesh(std::vector<Float3> &position, std::vector<Int4> &tets,
                         std::string tet_name, std::function<void(const std::vector<Float3> &, std::vector<bool> &is_fixed)> get_fixed_verts_func,
                         Float3 t, Float3 r, Float3 s, InputTetrahedralMesh &input_tet) {
    get_scene_params().simulate_tet = true;

    input_tet.mesh_name = (tet_name);

    input_tet.m_translation = (t);
    input_tet.m_rotation = (r);
    input_tet.m_scale = (s);
    input_tet.m_matrix = (make_model_matrix(t, r, s));

    // Prapare Surface Mat Data
    {
        // Get Surface Faces
        const uint num_verts = position.size();
        const uint num_tets = tets.size();

        uint num_surface_faces = 0;
        std::vector<Int3> curr_surface_faces;
        std::vector<uint> curr_outer_tets;
        std::vector<uint> curr_inner_tets;
        uint num_surface_edges = 0;
        std::vector<Int2> curr_surface_edges;
        std::vector<Int4> tmp_bending_edges;
        uint num_surface_verts = 0;
        std::vector<uint> curr_suface_verts;

        SimMesh::extract_surface_face_and_vert_from_tets(position, tets, curr_inner_tets, curr_outer_tets, curr_surface_faces, curr_suface_verts);

        // fast_format("Surface Vert Size = {}, Total Size = {}, LastIndices of Surface Verts = {}", curr_suface_verts.size(), num_verts, curr_suface_verts.back());

        {
            // Sort Tet By Surface
            const uint num_inner = curr_inner_tets.size();
            const uint num_outer = curr_outer_tets.size();
            if (num_inner + num_outer != num_tets) {
                fast_format_err("Sum of Inner Tets & Outer Tets Is Not Equal To Total Tets");
                exit(0);
            }

            std::vector<Int4> tets_copy(tets);
            parallel_for(0, num_outer, [&](const uint tid) {
                const uint index = curr_outer_tets[tid];
                tets[tid] = tets_copy[index];
            });
            parallel_for(0, num_inner, [&](const uint tid) {
                const uint index = curr_inner_tets[tid];
                tets[num_outer + tid] = tets_copy[index];
            });
        }

        SimMesh::extract_edges_from_surface<false>(curr_surface_faces, curr_surface_edges, tmp_bending_edges);

        input_tet.positions = (position);
        input_tet.tets = (tets);
        input_tet.inner_tets = (curr_inner_tets);
        input_tet.outer_tets = (curr_outer_tets);
        input_tet.surface_verts = (curr_suface_verts);
        input_tet.surface_faces = (curr_surface_faces);
        input_tet.surface_edges = (curr_surface_edges);// Actually To Be Filled
    }

    std::vector<bool> is_fixed(position.size(), false);
    get_fixed_verts_func(position, is_fixed);
    input_tet.fixed_points = (is_fixed);

    AABB aabb = parallel_for_and_reduce_sum<AABB>(0, position.size(), [&](const uint vid) {
        return AABB(position[vid]);
    });
    fast_format("Tetrahedral Info : NumVerts = {} , NumTets = {} , NumSurfaceVerts = {} , NumSurfaceFaces = {} , NumSurfaceTets = {} , NumSurfaceEdges = {} ",
                position.size(), tets.size(), input_tet.surface_verts.size(), input_tet.surface_faces.size(), input_tet.outer_tets.size(), input_tet.surface_edges.size());
}

void AppendTetrahedralModel(std::string model_name,
                            std::function<void(const std::vector<Float3> &local_position, std::vector<bool> &is_fixed)> get_fixed_verts_func,
                            Float3 translation,
                            Float3 rotation,
                            Float3 scale, bool use_default_path, InputTetrahedralMesh &input_tet) {
    std::vector<Float3> sa_position;
    std::vector<Int4> sa_tets;

    auto dot_pos = model_name.find_last_of('.');
    if (dot_pos == std::string::npos) {
        fast_format_err("Error: File extension not found in model name: {} ", model_name);
        return;
    }
    std::string extension = model_name.substr(dot_pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    std::string obj_name = model_name;
    {
        std::filesystem::path path(obj_name);
        obj_name = path.stem().string() + "_" + std::to_string(input_tet.tets.size());
    }

    bool read_result;
    if (extension == "t") {
        read_result = SimMesh::read_tet_file_t(model_name, sa_position, sa_tets, true);
        preprocess_tet_mesh(sa_position, sa_tets, obj_name, get_fixed_verts_func, translation, rotation, scale, input_tet);
    } else if (extension == "vtk") {
        read_result = SimMesh::read_tet_file_vtk(model_name, sa_position, sa_tets, true);
        preprocess_tet_mesh(sa_position, sa_tets, obj_name, get_fixed_verts_func, translation, rotation, scale, input_tet);
    } else {
        std::cerr << "Error: Unsupported file format: " << extension << std::endl;
        fast_format_err("Error: Unsupported file format:", extension);
        return;
    }
}
void AppendTriangleObstacleModel(std::string model_name,
                                 Float3 translation, Float3 rotation, Float3 scale, bool use_default_path,
                                 const std::map<uint, AnimationPerFrameData> &animation_info, InputTriangleMesh &input_mesh) {
    TriangleMeshData curr_mesh;
    bool second_read = SimMesh::read_mesh_file(model_name, curr_mesh, use_default_path);

    std::string obj_name = model_name;
    {
        std::filesystem::path path(obj_name);
        obj_name = path.stem().string();
    }

    input_mesh.mesh_name = (obj_name);
    input_mesh.m_translation = (translation);
    input_mesh.m_rotation = (rotation);
    input_mesh.m_scale = (scale);
    input_mesh.m_matrix = (make_model_matrix(translation, rotation, scale));
    input_mesh.mesh = (curr_mesh);
}

void init_tet_mesh(TetData *mesh_data) {

    InputTetrahedralMesh input_mesh;
    AppendTetrahedralModel(
        "SIGGRAPH.vtk", [](const std::vector<Float3> &local_position, std::vector<bool> &is_fixed) {}, makeFloat3(0.0f), makeFloat3(0.0f), makeFloat3(1.0f), true, input_mesh);

    const uint num_verts = input_mesh.positions.size();
    const uint num_faces = input_mesh.surface_faces.size();
    const uint num_edges = input_mesh.surface_edges.size();
    const uint num_tets = input_mesh.tets.size();

    fast_format("Tetrahedron : (numVerts : {}) (numFaces : {})  (numEdges : {}) "
                "(numTets : {})",
                num_verts, num_faces, num_edges, num_tets);

    // Constant scalar
    {
        mesh_data->num_verts_total = num_verts;
        mesh_data->num_surface_faces_total = num_faces;
        mesh_data->num_surface_edges_total = num_edges;
        mesh_data->num_tets_total = num_tets;
    }

    upload_from(mesh_data->sa_rest_position, input_mesh.positions);
    upload_from(mesh_data->sa_surface_faces, input_mesh.surface_faces);
    upload_from(mesh_data->sa_surface_edges, input_mesh.surface_edges);
    upload_from(mesh_data->sa_tets, input_mesh.tets);
    mesh_data->sa_rest_velocity.resize(num_verts);

    // Init vert info
    {
        // Set rest position & velocity
        {
            parallel_for(0, num_verts, [&](const uint vid) {
                Float3 model_position = mesh_data->sa_rest_position[vid];
                Float4x4 model_matrix = make_model_matrix(input_mesh.m_translation, input_mesh.m_rotation, input_mesh.m_scale);
                Float3 world_position = affine_position(model_matrix, model_position);
                mesh_data->sa_rest_position[vid] = world_position;
                mesh_data->sa_rest_velocity[vid] = makeFloat3(0.0f);
            });
        }

        // Set fixed-points
        {
            mesh_data->sa_is_fixed.resize(num_verts);
            parallel_for(0, num_verts, [&](const uint vid) {
                mesh_data->sa_is_fixed[vid] = false;
            });
        }

        // Set vert mass
        {
            mesh_data->sa_vert_mass.resize(num_verts);
            mesh_data->sa_vert_mass_inv.resize(num_verts);

            const float defulat_density = 0.01f;
            const float defulat_mass =
                defulat_density * get_scene_params().default_mass;
            parallel_for(0, num_verts, [&](const uint vid) {
                bool is_fixed = mesh_data->sa_is_fixed[vid] != 0;
                mesh_data->sa_vert_mass[vid] = (defulat_mass);
                mesh_data->sa_vert_mass_inv[vid] =
                    is_fixed ? 0.0f : 1.0f / (defulat_mass);
            });
        }
    }

    // Init adjacent list
    {
        mesh_data->vert_adj_tets.resize(num_verts);

        // Vert adj tets
        for (uint tid = 0; tid < num_tets; tid++) {
            auto tet = mesh_data->sa_tets[tid];
            for (uint j = 0; j < 4; j++)
                mesh_data->vert_adj_tets[tet[j]].push_back(tid);
        }
        upload_2d_csr_from(mesh_data->sa_vert_adj_tets_csr,
                           mesh_data->vert_adj_tets);

        // Vert adj verts based on 1-order connection
        mesh_data->vert_adj_verts.resize(num_verts);
        auto insert_adj_vert = [&](const uint vid, const uint adj_vid) {
            auto &list = mesh_data->vert_adj_verts[vid];
            if (std::find(list.begin(), list.end(), adj_vid) == list.end()) {
                list.push_back(adj_vid);
            }
        };
        for (uint tid = 0; tid < num_tets; tid++) {
            auto tet = mesh_data->sa_tets[tid];
            for (uint ii = 0; ii < 4; ii++) {
                const uint vid = tet[ii];
                for (uint jj = ii + 1; jj < 4; jj++) {
                    const uint adj_vid = tet[jj];
                    insert_adj_vert(vid, adj_vid);
                    insert_adj_vert(adj_vid, vid);
                }
            }
        }
        upload_2d_csr_from(mesh_data->sa_vert_adj_verts_csr, mesh_data->vert_adj_verts);
    }

    // Init energy
    {
        // Rest spring length
        mesh_data->sa_Dm.resize(num_edges);
        mesh_data->sa_Dm_inv.resize(num_edges);
        mesh_data->sa_tet_volumn.resize(num_edges);
        parallel_for(0, num_tets, [&](const uint tid) {
            Int4 tet = mesh_data->sa_tets[tid];
            Float3 vert_pos[4] = {
                mesh_data->sa_rest_position[tet[0]],
                mesh_data->sa_rest_position[tet[1]],
                mesh_data->sa_rest_position[tet[2]],
                mesh_data->sa_rest_position[tet[3]],
            };
            Float3x3 Dm = makeFloat3x3(
                vert_pos[1] - vert_pos[0],
                vert_pos[2] - vert_pos[0],
                vert_pos[3] - vert_pos[0]);
            Float3x3 Dm_inv = inverse_mat(Dm);
            mesh_data->sa_Dm[tid] = Dm;
            mesh_data->sa_Dm_inv[tid] = Dm_inv;
            mesh_data->sa_tet_volumn[tid] = compute_tet_volumn(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3]);
        });
    }

    // Init vert status
    {
    }
}
void init_obstacle_mesh(ObstacleData *mesh_data) {
    InputTriangleMesh input_mesh;
    AppendTriangleObstacleModel(
        "bowl.obj", makeFloat3(0.0f, 0.0f, 0.0f), makeFloat3(0.0f), makeFloat3(0.0f, 1.0f, 0.0f), true, {}, input_mesh);
    const uint num_verts = input_mesh.mesh.model_positions.size();
    const uint num_faces = input_mesh.mesh.faces.size();
    const uint num_edges = input_mesh.mesh.edges.size();
    fast_format("Obstacle Mesh : (numVerts : {}) (numFaces : {})  (numEdges : {}) ",
                num_verts, num_faces, num_edges);

    // Constant scalar
    {
        mesh_data->num_verts_total = num_verts;
        mesh_data->num_faces_total = num_faces;
        mesh_data->num_edges_total = num_edges;
    }

    upload_from(mesh_data->sa_rest_position, input_mesh.mesh.model_positions);
    upload_from(mesh_data->sa_rest_position, input_mesh.mesh.model_positions);
    upload_from(mesh_data->sa_faces, input_mesh.mesh.faces);
    upload_from(mesh_data->sa_edges, input_mesh.mesh.edges);
    mesh_data->sa_rest_velocity.resize(num_verts);

    // Set rest position & velocity
    {
        parallel_for(0, num_verts, [&](const uint vid) {
            Float3 model_position = mesh_data->sa_rest_position[vid];
            Float4x4 model_matrix = make_model_matrix(input_mesh.m_translation, input_mesh.m_rotation, input_mesh.m_scale);
            Float3 world_position = affine_position(model_matrix, model_position);
            mesh_data->sa_rest_position[vid] = world_position;
            mesh_data->sa_rest_velocity[vid] = makeFloat3(0.0f);
        });
    }
}
void init_xpbd_data(TetData *mesh_data, ObstacleData *obstacle_data, XpbdData *xpbd_data) {
    // To Be Done
    xpbd_data->resize(mesh_data, obstacle_data);
}

class CpuSolver {
public:
    CpuSolver() {}
    ~CpuSolver() {}

    void get_data_pointer(XpbdData *xpbd_data,
                          TetData *mesh_data,
                          ObstacleData *obstacle_data,
                          VivaceColoringData *coloring_data,
                          LbvhFaceEdgeData *lbvh_data_obstacle,
                          LbvhFaceEdgeData *lbvh_data_tet,
                          XpbdSelfCollision *self_collision_data_tet,
                          XpbdObstacleCollision *obstacle_collision_data_tet) {
        this->xpbd_data = xpbd_data;
        this->mesh_data = mesh_data;
        this->obstacle_data = obstacle_data;
        this->coloring_data = coloring_data;
        this->lbvh_data_obstacle = lbvh_data_obstacle;
        this->lbvh_data_tet = lbvh_data_tet;
        this->self_collision_data_tet = self_collision_data_tet;
        this->obstacle_collision_data_tet = obstacle_collision_data_tet;
    }
    void init_xpbd_system();
    static void init_simulation_params();

public:
    void physics_step_vbd();
    void physics_step_xpbd();
    // void physics_step_vbd_async();
    void fn_dispatch(const Launcher::LaunchParam &param);
    void compute_energy(const Buffer<Float3> &curr_cloth_position);
    void solve_constraints_XPBD();

private:
    void collision_detection();
    void predict_position();
    void update_velocity();
    void reset_constrains();
    void reset_collision_constrains();

private:
    void solve_constraint_tet_stress(Buffer<Float3> &curr_cloth_position, const uint cluster_idx);
    void solve_constraint_self_collision(Buffer<Float3> &curr_cloth_position, const uint cluster_idx);
    void solve_constraint_ground_collision(Buffer<Float3> &curr_cloth_position);
    void solve_constraint_obstacle_collision(Buffer<Float3> &curr_cloth_position);

private:
    XpbdData *xpbd_data;
    TetData *mesh_data;
    ObstacleData *obstacle_data;
    VivaceColoringData *coloring_data;
    LbvhFaceEdgeData *lbvh_data_obstacle;
    LbvhFaceEdgeData *lbvh_data_tet;

    LbvhFaceEdge<LBVHUpdateTypeObstacle> *lbvh_obstacle;
    LbvhFaceEdge<LBVHUpdateTypeCloth> *lbvh_tet;

    XpbdSelfCollision *self_collision_data_tet;
    XpbdObstacleCollision *obstacle_collision_data_tet;
};
class GpuSolver {
public:
    GpuSolver() {}
    ~GpuSolver() {}

    void get_data_pointer(XpbdData *xpbd_data,
                          TetData *mesh_data,
                          ObstacleData *obstacle_data,
                          VivaceColoringData *coloring_data,
                          LbvhFaceEdgeData *lbvh_data_obstacle,
                          LbvhFaceEdgeData *lbvh_data_tet,
                          XpbdSelfCollision *self_collision_data_tet,
                          XpbdObstacleCollision *obstacle_collision_data_tet) {
        this->xpbd_data = xpbd_data;
        this->mesh_data = mesh_data;
        this->obstacle_data = obstacle_data;
        this->coloring_data = coloring_data;
        this->lbvh_data_obstacle = lbvh_data_obstacle;
        this->lbvh_data_tet = lbvh_data_tet;
        this->self_collision_data_tet = self_collision_data_tet;
        this->obstacle_collision_data_tet = obstacle_collision_data_tet;
    }

    void init_xpbd_system();

public:
    void physics_step_vbd();
    void physics_step_xpbd();
    void physics_step_vbd_async();
    void register_dag(Launcher::Scheduler &scheduler);
    void evaluate_compuatation_matrix(Launcher::Scheduler &scheduler);
    void fn_dispatch(const Launcher::LaunchParam &param);
    void compute_energy(const Buffer<Float3> &curr_cloth_position);
    void solve_constraints_XPBD();

private:
    void collision_detection();
    void predict_position();
    void update_velocity();
    void reset_constrains();
    void reset_collision_constrains();

private:
    void solve_constraint_tet_stress(Buffer<Float3> &curr_cloth_position, const uint cluster_idx);
    void solve_constraint_self_collision(Buffer<Float3> &curr_cloth_position, const uint cluster_idx);
    void solve_constraint_ground_collision(Buffer<Float3> &curr_cloth_position);
    void solve_constraint_obstacle_collision(Buffer<Float3> &curr_cloth_position);

private:
    void vbd_evaluate_inertia(Buffer<Float3> &curr_cloth_position,
                              const uint cluster_idx);
    void vbd_evaluate_stretch_spring(Buffer<Float3> &curr_cloth_position,
                                     const uint cluster_idx);
    void vbd_evaluate_bending(Buffer<Float3> &curr_cloth_position,
                              const uint cluster_idx);
    void vbd_step(Buffer<Float3> &curr_cloth_position, const uint cluster_idx);

private:
    XpbdData *xpbd_data;
    TetData *mesh_data;
    ObstacleData *obstacle_data;
    VivaceColoringData *coloring_data;
    LbvhFaceEdgeData *lbvh_data_obstacle;
    LbvhFaceEdgeData *lbvh_data_tet;
    XpbdSelfCollision *self_collision_data_tet;
    XpbdObstacleCollision *obstacle_collision_data_tet;

    LbvhFaceEdge<LBVHUpdateTypeObstacle> *lbvh_obstacle;
    LbvhFaceEdge<LBVHUpdateTypeCloth> *lbvh_tet;

    CpuSolver *cpu_solver;


private:
    gpuFunction fn_empty;
    gpuFunction fn_reset_bool;
    gpuFunction fn_reset_uint;
    gpuFunction fn_reset_float;
    gpuFunction fn_copy_from_A_to_B;
    gpuFunction fn_copy_from_A_to_B_and_C;
    gpuFunction fn_read_and_solve_conflict;

    gpuFunction fn_predict_position;
    gpuFunction fn_update_velocity;

    gpuFunction fn_xpbd_constraint_neohookean;
    gpuFunction fn_xpbd_constraint_ground_collision;

    gpuFunction fn_compute_energy_inertia;
    gpuFunction fn_compute_energy_stress;
    gpuFunction fn_compute_energy_collision_vf;
    gpuFunction fn_compute_energy_collision_vv;
    gpuFunction fn_test_sum;
    gpuFunction fn_test_sum_2;
};
static uint energy_idx = 0;

void CpuSolver::init_xpbd_system() {

    xpbd_data->sa_system_energy.resize(10240);

    const uint num_verts = mesh_data->num_verts_total;
    xpbd_data->sa_x_tilde.resize(num_verts);
    xpbd_data->sa_x.resize(num_verts);
    xpbd_data->sa_v.resize(num_verts);
    xpbd_data->sa_x_iter_start.resize(num_verts);
    xpbd_data->sa_x_step_start.resize(num_verts);

    xpbd_data->sa_v.set_zero();

    for (auto &buffer : xpbd_data->sa_async_iter_positions_tet)
        buffer.resize(mesh_data->num_verts_total);
    for (auto &buffer : xpbd_data->sa_async_begin_positions_tet)
        buffer.resize(mesh_data->num_verts_total);

    // Init Constraints
    {
        xpbd_data->lambda_tet_stress_deviatoric_term.resize(
            mesh_data->num_tets_total);
        xpbd_data->lambda_tet_stress_hydrostatic_term.resize(
            mesh_data->num_tets_total);
    }

    // Graph Coloring
    std::vector<std::vector<uint>> tmp_clusterd_constraint_tet_stress;
    {
        auto fn_graph_coloring_sequenced_constraint =
            [](const uint num_elements, const std::string &constraint_name,
               std::vector<std::vector<uint>> &clusterd_constraint,
               const std::vector<std::vector<uint>> &vert_adj_elements,
               const auto &element_indices, const uint nv) {
                std::vector<bool> marked_constrains(num_elements, false);
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

                while (true) {
                    std::vector<uint> current_cluster;
                    std::vector<bool> current_marked(marked_constrains);
                    for (uint id = 0; id < num_elements; id++) {
                        if (current_marked[id]) {
                            continue;
                        } else {
                            // Add To Sets
                            marked_constrains[id] = true;
                            current_cluster.push_back(id);

                            // Mark
                            current_marked[id] = true;
                            auto element = element_indices[id];
                            for (uint j = 0; j < nv; j++) {
                                for (const uint &adj_eid : vert_adj_elements[element[j]]) {
                                    current_marked[adj_eid] = true;
                                }
                            }
                        }
                    }

                    const uint cluster_size = static_cast<uint>(current_cluster.size());
                    total_marked_count += cluster_size;

                    if (merge_small_cluster && cluster_size < color_threashold) {
                        rest_cluster.insert(rest_cluster.end(), current_cluster.begin(),
                                            current_cluster.end());
                    } else {
                        clusterd_constraint.push_back(current_cluster);
                    }

                    if (total_marked_count == num_elements)
                        break;
                }

                if (merge_small_cluster && !rest_cluster.empty()) {
                    clusterd_constraint.push_back(rest_cluster);
                }

                fast_format("Cluster Count of {} = {}", constraint_name,
                            clusterd_constraint.size());
            };

        auto fn_get_prefix =
            [](auto &prefix_buffer,
               const std::vector<std::vector<uint>> &clusterd_constraint) {
                const uint num_cluster = clusterd_constraint.size();
                prefix_buffer.resize(num_cluster + 1);
                uint prefix = 0;
                for (uint cluster = 0; cluster < num_cluster; cluster++) {
                    prefix_buffer[cluster] = prefix;
                    prefix += clusterd_constraint[cluster].size();
                }
                prefix_buffer[num_cluster] = prefix;
            };

        fn_graph_coloring_sequenced_constraint(
            mesh_data->num_tets_total, "NeoHook Stress Constraint",
            tmp_clusterd_constraint_tet_stress, mesh_data->vert_adj_tets,
            mesh_data->sa_tets, 4);

        xpbd_data->num_clusters_tet_stress =
            tmp_clusterd_constraint_tet_stress.size();

        fn_get_prefix(xpbd_data->prefix_tet_stress,
                      tmp_clusterd_constraint_tet_stress);

        upload_2d_csr_from(xpbd_data->clusterd_constraint_tet_stress,
                           tmp_clusterd_constraint_tet_stress);
    }

    // Precomputation
    {
        // Tet Stress Constraint
        {
            xpbd_data->sa_merged_tets.resize(mesh_data->num_tets_total);
            xpbd_data->sa_merged_tet_volumn.resize(mesh_data->num_tets_total);
            xpbd_data->sa_merged_Dm_inv.resize(
                mesh_data->num_tets_total);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_tet_stress.size();
                 cluster++) {
                const auto &curr_cluster = tmp_clusterd_constraint_tet_stress[cluster];
                parallel_for(0, curr_cluster.size(), [&](const uint i) {
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_tets[prefix + i] =
                            mesh_data->sa_tets[eid];
                        xpbd_data->sa_merged_tet_volumn[prefix + i] =
                            mesh_data->sa_tet_volumn[eid];
                        xpbd_data->sa_merged_Dm_inv[prefix + i] =
                            mesh_data->sa_Dm_inv[eid];
                    }
                });
                prefix += curr_cluster.size();
            }
            if (prefix != mesh_data->num_tets_total) {
                fast_format_err("Sum of Bending Cluster Is Not Equal Than Orig");
            }
        }
    }
}
void GpuSolver::init_xpbd_system() {
    {
        NS::Error *err;

        std::string full_path_xpbd = std::string(SELF_RESOURCES_PATH) +
                                     std::string("/metal_libs/") +
                                     std::string("example3.metallib");
        MTL::Library *library_xpbd =
            get_device()->newLibrary(std_string_to_ns_string(full_path_xpbd), &err);
        check_err(library_xpbd, err);

        fn_empty.load(library_xpbd, "empty");
        fn_reset_bool.load(library_xpbd, "reset_bool");
        fn_reset_uint.load(library_xpbd, "reset_uint");
        fn_reset_float.load(library_xpbd, "reset_float");

        fn_copy_from_A_to_B.load(library_xpbd, "copy_from_A_to_B");
        fn_copy_from_A_to_B_and_C.load(library_xpbd, "copy_from_A_to_B_and_C");
        fn_read_and_solve_conflict.load(library_xpbd, "read_and_solve_conflict");

        fn_predict_position.load(library_xpbd, "predict_position");
        fn_update_velocity.load(library_xpbd, "update_velocity");

        fn_xpbd_constraint_neohookean.load(library_xpbd, "constraint_neohookean");
        fn_xpbd_constraint_ground_collision.load(library_xpbd, "constraint_ground_collision");

        fn_compute_energy_inertia.load(library_xpbd, "compute_energy_inertia");
        fn_compute_energy_stress.load(
            library_xpbd, "compute_energy_stress_neohookean");
        fn_compute_energy_collision_vv.load(library_xpbd, "compute_energy_collision_vv");
        fn_compute_energy_collision_vf.load(library_xpbd, "compute_energy_collision_vf");
        fn_test_sum.load(library_xpbd, "test_sum");
        fn_test_sum_2.load(library_xpbd, "test_sum_2");
    }

    // Init LBVH and Vivace
    {
    }
}

void CpuSolver::reset_constrains() {
    auto fn_reset_template = [&](Buffer<float> &buffer) {
        parallel_set(buffer.data(), buffer.size(), 0.0f);
    };

    fn_reset_template(xpbd_data->lambda_tet_stress_deviatoric_term);
    fn_reset_template(xpbd_data->lambda_tet_stress_hydrostatic_term);
}
void GpuSolver::reset_constrains() {
    auto fn_reset_template = [&](Buffer<float> &buffer) {
        get_command_list().add_task(fn_reset_float);
        fn_reset_float.bind_ptr(buffer);
        fn_reset_float.launch_async(buffer.size());
    };

    fn_reset_template(xpbd_data->lambda_tet_stress_deviatoric_term);
    fn_reset_template(xpbd_data->lambda_tet_stress_hydrostatic_term);
}

void CpuSolver::reset_collision_constrains() {}
void GpuSolver::reset_collision_constrains() {}

void CpuSolver::init_simulation_params() {
    get_scene_params().print_cost_detail = true;
    get_scene_params().print_xpbd_convergence = false;// false true

    if (get_scene_params().use_substep) {
        get_scene_params().implicit_dt = 1.f / 60.f;
    } else {
        get_scene_params().num_substep = 1;
        get_scene_params().constraint_iter_count = 200;
    }

    if (get_scene_params().use_small_timestep) {
        get_scene_params().implicit_dt = 0.001f;
    }

    get_scene_params().use_multi_buffer = false;
    get_scene_params().num_iteration =
        get_scene_params().num_substep * get_scene_params().constraint_iter_count;
    get_scene_params().collision_detection_frequece = 1;

    get_scene_params().stiffness_stretch_BaraffWitkin =
        FEM::calcSecondLame(get_scene_params().youngs_modulus_cloth,
                            get_scene_params().poisson_ratio_cloth);// mu;
    get_scene_params().stiffness_stretch_spring =
        FEM::calcSecondLame(get_scene_params().youngs_modulus_cloth,
                            get_scene_params().poisson_ratio_cloth);// mu;
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

void CpuSolver::collision_detection() {
    // TODO
}
void GpuSolver::collision_detection() {
    // TODO
}

void CpuSolver::predict_position() {
    parallel_for(0, mesh_data->num_verts_total, [&](const uint vid) {
        Constrains::Core::predict_position(
            vid, xpbd_data->sa_x.data(), xpbd_data->sa_v.data(),
            xpbd_data->sa_x_step_start.data(), xpbd_data->sa_x_tilde.data(), false,
            nullptr, mesh_data->sa_vert_mass.data(), mesh_data->sa_is_fixed.data(),
            get_scene_params().get_substep_dt(), false);
    });
}
void GpuSolver::predict_position() {
    get_command_list().add_task(fn_predict_position);
    fn_predict_position.bind_ptr(xpbd_data->sa_x);
    fn_predict_position.bind_ptr(xpbd_data->sa_v);
    fn_predict_position.bind_ptr(xpbd_data->sa_x_step_start);
    fn_predict_position.bind_ptr(xpbd_data->sa_x_tilde);
    fn_predict_position.bind_constant(false);
    fn_predict_position.bind_ptr(xpbd_data->sa_x);
    fn_predict_position.bind_ptr(mesh_data->sa_vert_mass);
    fn_predict_position.bind_ptr(mesh_data->sa_is_fixed);
    fn_predict_position.bind_constant(get_scene_params().get_substep_dt());
    fn_predict_position.bind_constant(false);
    fn_predict_position.launch_async(mesh_data->num_verts_total);
}

void CpuSolver::update_velocity() {
    parallel_for(0, mesh_data->num_verts_total, [&](const uint vid) {
        Constrains::Core::update_velocity(
            vid, xpbd_data->sa_v.data(), xpbd_data->sa_x.data(),
            xpbd_data->sa_x_iter_start.data(), xpbd_data->sa_x_step_start.data(),
            xpbd_data->sa_v.data(), get_scene_params().get_substep_dt(),
            get_scene_params().damping_cloth, false);
    });
}
void GpuSolver::update_velocity() {
    get_command_list().add_task(fn_update_velocity);
    fn_update_velocity.bind_ptr(xpbd_data->sa_v);
    fn_update_velocity.bind_ptr(xpbd_data->sa_x);
    fn_update_velocity.bind_ptr(xpbd_data->sa_x_iter_start);
    fn_update_velocity.bind_ptr(xpbd_data->sa_x_step_start);
    fn_update_velocity.bind_ptr(xpbd_data->sa_v);
    fn_update_velocity.bind_constant(get_scene_params().get_substep_dt());
    fn_update_velocity.bind_constant(get_scene_params().damping_cloth);
    fn_update_velocity.bind_constant(false);
    fn_update_velocity.launch_async(mesh_data->num_verts_total);
}

void CpuSolver::compute_energy(const Buffer<Float3> &curr_position) {
    return;

    if (!get_scene_params().print_xpbd_convergence)
        return;
    // fast_format("buffer size = {}", curr_position.size());
    // fast_format("CPU Call {}", energy_idx);

    double energy = 0.0;
    double energy_inertia = 0.f, energy_stretch = 0.f, energy_bending = 0.f;

    // Inertia
    {
        energy_inertia = parallel_for_and_reduce_sum<double>(
            0, mesh_data->num_verts_total, [&](const uint vid) {
                return Constrains::Energy::compute_energy_inertia(
                    vid, curr_position.data(), &get_scene_params(),
                    mesh_data->sa_is_fixed.data(), mesh_data->sa_vert_mass.data(),
                    xpbd_data->sa_x_tilde.data());
            });
    }

    // Stretch
    {
        auto lambda = FEM::calcFirstLame(get_scene_params().youngs_modulus_tet, get_scene_params().poisson_ratio_tet);
        auto mu = FEM::calcSecondLame(get_scene_params().youngs_modulus_tet, get_scene_params().poisson_ratio_tet);
        energy_stretch = parallel_for_and_reduce_sum<double>(
            0, mesh_data->num_tets_total, [&](const uint tid) {
                return Constrains::Energy::compute_energy_stress_neohookean(tid,
                                                                            curr_position.data(),
                                                                            mesh_data->sa_tets.data(),
                                                                            mesh_data->sa_Dm_inv.data(),
                                                                            mesh_data->sa_tet_volumn.data(),
                                                                            lambda, mu);
            });
    }

    // Ground Collision
    float energy_ground_collision = 0.0f;
    if (get_scene_params().use_floor) {
        energy_ground_collision += parallel_for_and_reduce_sum<float>(0,
                                                                      mesh_data->num_verts_total, [&](const uint vid) {
                                                                          return Constrains::Energy::compute_energy_collision_ground(vid,
                                                                                                                                     curr_position.data(),
                                                                                                                                     get_scene_params().xpbd_stiffness_collision,
                                                                                                                                     get_scene_params().thickness_vv_obstacle);
                                                                      });
    }

    // Obstacle Collisoin
    float energy_obs_collision = 0.0f;
    // if (get_scene_params().use_obstacle_collision)
    // {
    //     const auto& obstacle_collision_data = obstacle_collision_data_cloth;
    //     const float thickness1 = 0.0f;
    //     const float thickness2 = get_scene_params().thickness_vv_obstacle;
    //     energy_obs_collision += parallel_for_and_reduce_sum<float>(0,
    //     obstacle_collision_data->collision_count[0], [&](const uint i)
    //     {
    //         return Constrains::Energy::compute_energy_collision_vf(i,
    //         curr_position.data(), obstacle_data->sa_substep_position.data(),
    //         obstacle_collision_data->narrow_phase_list_pair_vf.data(),
    //         obstacle_collision_data->collision_count.data(), thickness2);
    //     });
    // }

    // Self Collision
    float energy_self_collision = 0.0f;
    // if (get_scene_params().use_self_collision)
    // {
    //     const auto& self_collision_data = self_collision_data_cloth;
    //     const float thickness1 = 0.0f;
    //     const float thickness2 = get_scene_params().thickness_vv_cloth;
    //     energy_self_collision = parallel_for_and_reduce_sum<float>(0,
    //     self_collision_data->collision_count[0], [&](const uint i)
    //     {
    //         return Constrains::Energy::compute_energy_collision_vv(i,
    //         curr_position.data(),
    //         self_collision_data->narrow_phase_list_pair_vv.data(),
    //         self_collision_data->collision_count.data(), thickness2);
    //     });
    // }

    double total_energy = energy_inertia + energy_stretch + energy_bending +
                          energy_ground_collision + energy_obs_collision + energy_self_collision;

    xpbd_data->sa_system_energy[energy_idx++] = total_energy;
}
void GpuSolver::compute_energy(const Buffer<Float3> &curr_position) {
    // return;

    if (!get_scene_params().print_xpbd_convergence)
        return;
    // fast_format("buffer size = {}", curr_position.size());
    // fast_format("GPU Call {}", energy_idx);

    // get_command_list().send_and_wait();
    // cpu_solver->compute_energy(curr_position);
    // return;

    // Inertia
    {
        get_command_list().add_task(fn_compute_energy_inertia);
        fn_compute_energy_inertia.bind_ptr(xpbd_data->sa_system_energy);
        fn_compute_energy_inertia.bind_constant(energy_idx);
        fn_compute_energy_inertia.bind_ptr(curr_position);

        fn_compute_energy_inertia.bind_ptr(get_scene_params_array());
        fn_compute_energy_inertia.bind_ptr(mesh_data->sa_is_fixed);
        fn_compute_energy_inertia.bind_ptr(mesh_data->sa_vert_mass);
        fn_compute_energy_inertia.bind_ptr(xpbd_data->sa_x_tilde);

        fn_compute_energy_inertia.launch_async(mesh_data->num_verts_total);
    }

    // Stress
    {
        const float m_first_lame = FEM::calcFirstLame(get_scene_params().youngs_modulus_tet, get_scene_params().poisson_ratio_tet);  // lambda
        const float m_second_lame = FEM::calcSecondLame(get_scene_params().youngs_modulus_tet, get_scene_params().poisson_ratio_tet);// mu
        {
            get_command_list().add_task(fn_compute_energy_stress);
            fn_compute_energy_stress.bind_ptr(xpbd_data->sa_system_energy);
            fn_compute_energy_stress.bind_constant(energy_idx);
            fn_compute_energy_stress.bind_ptr(curr_position);

            fn_compute_energy_stress.bind_ptr(mesh_data->sa_tets);
            fn_compute_energy_stress.bind_ptr(mesh_data->sa_Dm_inv);
            fn_compute_energy_stress.bind_ptr(mesh_data->sa_tet_volumn);
            fn_compute_energy_stress.bind_constant(m_first_lame);
            fn_compute_energy_stress.bind_constant(m_second_lame);

            fn_compute_energy_stress.launch_async(mesh_data->num_tets_total);
        }
    }
    if (get_scene_params().use_obstacle_collision) {
        const auto &obstacle_collision_data = obstacle_collision_data_tet;
        const float thickness1 = 0.0f;
        const float thickness2 = get_scene_params().thickness_vv_obstacle;

        get_command_list().add_task(fn_compute_energy_collision_vf);
        fn_compute_energy_collision_vf.bind_ptr(xpbd_data->sa_system_energy);
        fn_compute_energy_collision_vf.bind_constant(energy_idx);
        fn_compute_energy_collision_vf.bind_ptr(curr_position);
        fn_compute_energy_collision_vf.bind_ptr(obstacle_data->sa_substep_position);

        fn_compute_energy_collision_vf.bind_ptr(obstacle_collision_data->narrow_phase_list_pair_vf);
        fn_compute_energy_collision_vf.bind_ptr(obstacle_collision_data->collision_count);
        fn_compute_energy_collision_vf.bind_constant(thickness2);
        fn_compute_energy_collision_vf.launch_async(obstacle_collision_data->obstacle_collision_indirect_cmd_buffer, 0);
    }

    if (get_scene_params().use_self_collision) {
    }

    energy_idx++;
}

// XPBD constraints

void CpuSolver::solve_constraint_tet_stress(Buffer<Float3> &sa_iter_position, const uint cluster_idx) {
    const uint curr_prefix = xpbd_data->prefix_tet_stress[cluster_idx];
    const uint next_prefix = xpbd_data->prefix_tet_stress[cluster_idx + 1];
    const uint num_elements_clustered = next_prefix - curr_prefix;

    const float m_first_lame = FEM::calcFirstLame(get_scene_params().youngs_modulus_tet, get_scene_params().poisson_ratio_tet);  // lambda
    const float m_second_lame = FEM::calcSecondLame(get_scene_params().youngs_modulus_tet, get_scene_params().poisson_ratio_tet);// mu

    parallel_for(
        0, num_elements_clustered, [&](const uint i) {
            const uint tet_id = curr_prefix + i;
            Constrains::solve_tetrahedral_fem_NeoHookean_template(
                tet_id,
                sa_iter_position.data(), sa_iter_position.data(), xpbd_data->sa_x_step_start.data(),
                nullptr,
                xpbd_data->lambda_tet_stress_hydrostatic_term.data(), xpbd_data->lambda_tet_stress_deviatoric_term.data(),
                mesh_data->sa_vert_mass_inv.data(),
                xpbd_data->sa_merged_tets.data(), xpbd_data->sa_merged_tet_volumn.data(), xpbd_data->sa_merged_Dm_inv.data(),
                m_first_lame, m_second_lame,
                get_scene_params().get_substep_dt(), false);
        },
        32);
}
void GpuSolver::solve_constraint_tet_stress(Buffer<Float3> &sa_iter_position, const uint cluster_idx) {
}
void CpuSolver::solve_constraint_ground_collision(Buffer<Float3> &sa_iter_position) {
    parallel_for(
        0, mesh_data->num_verts_total, [&](const uint vid) {
            Constrains::solve_ground_collision_template(vid,
                                                        &get_scene_params(),
                                                        sa_iter_position.data(), xpbd_data->sa_x_step_start.data(),
                                                        xpbd_data->lambda_ground_collision_tet.data(),
                                                        mesh_data->sa_vert_mass_inv.data());
        },
        32);
}
void GpuSolver::solve_constraint_ground_collision(Buffer<Float3> &sa_iter_position) {
}
void CpuSolver::solve_constraint_obstacle_collision(Buffer<Float3> &sa_iter_position) {
    parallel_for(
        0, mesh_data->num_surface_verts_total, [&](const uint surface_id) {
            Constrains::solve_obstacle_collision_vf_template_tet(surface_id,
                                                                 nullptr, sa_iter_position.data(),
                                                                 obstacle_data->sa_substep_position.data(), obstacle_data->sa_vert_velocity.data(),
                                                                 nullptr, sa_iter_position.data(),
                                                                 nullptr, xpbd_data->sa_x_step_start.data(),

                                                                 mesh_data->sa_surface_verts.data(),
                                                                 nullptr, mesh_data->sa_vert_mass_inv.data(),
                                                                 nullptr, mesh_data->sa_vert_mutex.data(),

                                                                 obstacle_collision_data_tet->vert_VV_num_narrow_phase.data(), obstacle_collision_data_tet->vert_VV_prefix_narrow_phase.data(),
                                                                 obstacle_collision_data_tet->vert_adj_elements.data(), obstacle_collision_data_tet->narrow_phase_list_pair_vf.data(),
                                                                 xpbd_data->lambda_sdf_collision_tet.data(), xpbd_data->lambda_sdf_collision_tet_friction.data(),
                                                                 get_scene_params().max_vf_per_vert_narrow_obstacle_collision,
                                                                 get_scene_params().thickness_vv_obstacle, get_scene_params().get_substep_dt(),
                                                                 get_scene_params().xpbd_stiffness_collision, get_scene_params().friction_obstacle_tet,
                                                                 0);
        },
        32);
}
void GpuSolver::solve_constraint_obstacle_collision(Buffer<Float3> &sa_iter_position) {
}
void CpuSolver::solve_constraint_self_collision(Buffer<Float3> &sa_iter_position, const uint cluster_idx) {

    const float thickness = get_scene_params().thickness_vv_tet;

    auto fn_self_collision_solver_per_element_vv_template = [&](const uint pair_idx) {
        Constrains::solve_self_collision_vv_per_collision_pair_template_tet(pair_idx,
                                                                            nullptr, xpbd_data->sa_x_step_start.data(),
                                                                            nullptr, sa_iter_position.data(),
                                                                            nullptr, sa_iter_position.data(),

                                                                            mesh_data->sa_surface_verts.data(),
                                                                            nullptr, mesh_data->sa_vert_mass_inv.data(),
                                                                            nullptr, mesh_data->sa_vert_mutex.data(),

                                                                            self_collision_data_tet->narrow_phase_list_pair_vv_merged.data(),
                                                                            xpbd_data->lambda_self_collision_tet.data(), xpbd_data->lambda_self_collision_friction_tet.data(),

                                                                            get_scene_params().get_substep_dt(), false, thickness,
                                                                            get_scene_params().xpbd_stiffness_collision, get_scene_params().friction_tet, 0);
    };

    // {
    //     const uint curr_prefix = vivace_data_tet->cluster_prefix[cluster_idx];
    //     const uint num_elements_clustered = vivace_data_tet->num_verts_in_cluster[cluster_idx];
    //     const uint *cluster = vivace_data_tet->clusterd_constraint_self_collision.data() + curr_prefix;

    //     if (num_elements_clustered == 0) return;

    //     parallel_for(
    //         0, num_elements_clustered, [&](const uint i) {
    //             const uint pair_idx = curr_prefix + i;
    //             fn_self_collision_solver_per_element_vv_template(pair_idx);
    //         },
    //         32);
    // }
}
void GpuSolver::solve_constraint_self_collision(Buffer<Float3> &sa_iter_position, const uint cluster_idx) {

    const float thickness = get_scene_params().thickness_vv_tet;

    auto fn_self_collision_solver_per_element_vv_template = [&](const uint pair_idx) {
        Constrains::solve_self_collision_vv_per_collision_pair_template_tet(pair_idx,
                                                                            nullptr, xpbd_data->sa_x_step_start.data(),
                                                                            nullptr, sa_iter_position.data(),
                                                                            nullptr, sa_iter_position.data(),

                                                                            mesh_data->sa_surface_verts.data(),
                                                                            nullptr, mesh_data->sa_vert_mass_inv.data(),
                                                                            nullptr, mesh_data->sa_vert_mutex.data(),

                                                                            self_collision_data_tet->narrow_phase_list_pair_vv_merged.data(),
                                                                            xpbd_data->lambda_self_collision_tet.data(), xpbd_data->lambda_self_collision_friction_tet.data(),

                                                                            get_scene_params().get_substep_dt(), false, thickness,
                                                                            get_scene_params().xpbd_stiffness_collision, get_scene_params().friction_tet, 0);
    };

    // {
    //     const uint curr_prefix = vivace_data_tet->cluster_prefix[cluster_idx];
    //     const uint num_elements_clustered = vivace_data_tet->num_verts_in_cluster[cluster_idx];
    //     const uint *cluster = vivace_data_tet->clusterd_constraint_self_collision.data() + curr_prefix;

    //     if (num_elements_clustered == 0) return;

    //     parallel_for(
    //         0, num_elements_clustered, [&](const uint i) {
    //             const uint pair_idx = curr_prefix + i;
    //             fn_self_collision_solver_per_element_vv_template(pair_idx);
    //         },
    //         32);
    // }
}

void CpuSolver::physics_step_xpbd() {
    xpbd_data->sa_x_step_start = xpbd_data->sa_x_frame;
    xpbd_data->sa_x = xpbd_data->sa_x_frame;
    xpbd_data->sa_v = xpbd_data->sa_v_frame;

    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;

    xpbd_data->sa_system_energy.set_zero();
    energy_idx = 0;

    SimClock clock;
    clock.start_clock();

    for (uint substep = 0; substep < num_substep; substep++)// 1 or 50 ?
    {
        {
            get_scene_params().current_substep = substep;
        }

        // SimClock substep_clock; substep_clock.start_clock();
        {
            predict_position();

            collision_detection();

            // Constraint iteration part
            {
                for (uint iter = 0; iter < constraint_iter_count; iter++)// 200 or 1 ?
                {
                    {
                        get_scene_params().current_it = iter;
                    }
                    solve_constraints_XPBD();
                }
            }

            update_velocity();
        }
        // substep_clock.end_clock();
    }
    float frame_cost = clock.end_clock();

    fast_format("   In Frame {} : CPU Cost = {:6.3f}",
                get_scene_params().current_frame, clock.duration());

    {
        if (get_scene_params().print_xpbd_convergence) {
            std::vector<double> list_energy(energy_idx);
            for (uint it = 0; it < list_energy.size(); it++) {
                list_energy[it] = xpbd_data->sa_system_energy[it];
            }
            fast_print_iterator(list_energy, "Energy Convergence");
            energy_idx = 0;
        }
    }

    xpbd_data->sa_x_frame = xpbd_data->sa_x;
    xpbd_data->sa_v_frame = xpbd_data->sa_v;
}
void GpuSolver::physics_step_xpbd() {
    xpbd_data->sa_x_step_start = xpbd_data->sa_x_frame;
    xpbd_data->sa_x = xpbd_data->sa_x_frame;
    xpbd_data->sa_v = xpbd_data->sa_v_frame;

    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;

    xpbd_data->sa_system_energy.set_zero();
    energy_idx = 0;

    SimClock clock;
    clock.start_clock();

    for (uint substep = 0; substep < num_substep; substep++)// 1 or 50 ?
    {
        {
            get_scene_params().current_substep = substep;
        }

        // SimClock substep_clock; substep_clock.start_clock();
        {
            predict_position();

            collision_detection();

            // Constraint iteration part
            {
                for (uint iter = 0; iter < constraint_iter_count; iter++)// 200 or 1 ?
                {
                    {
                        get_scene_params().current_it = iter;
                    }
                    if (get_scene_params().use_vbd_solver) {
                        solve_constraints_XPBD();
                    } else {
                        fast_format_err("empty solver");
                    }
                }
            }

            update_velocity();
        }
        // substep_clock.end_clock();
    }
    get_command_list().send_and_wait();///////// GPU need to wait

    float frame_cost = clock.end_clock();

    fast_format("   In Frame {} : GPU Cost = {:6.3f}",
                get_scene_params().current_frame, clock.duration());

    {
        if (get_scene_params().print_xpbd_convergence) {
            std::vector<double> list_energy(energy_idx);
            for (uint it = 0; it < list_energy.size(); it++) {
                list_energy[it] = xpbd_data->sa_system_energy[it];
            }
            fast_print_iterator(list_energy, "Energy Convergence");
            energy_idx = 0;
        }
    }

    xpbd_data->sa_x_frame = xpbd_data->sa_x;
    xpbd_data->sa_v_frame = xpbd_data->sa_v;
}

void CpuSolver::fn_dispatch(const Launcher::LaunchParam &param) {
    // return;
    // fast_format("CPU dispatch {} {}",
    // Launcher::taskNames.at(param.function_id), param.cluster_idx);

    // Asynchronous iteration part
    constexpr uint max_buffer_count = 32;
    constexpr bool print_buffer_idx = false;
    auto fn_get_iter_buffer = [&](const uint buffer_idx) -> Buffer<Float3> & {
        // if constexpr (print_buffer_idx) fast_format("Iter buffer {} ({}) size =
        // {}", buffer_idx, buffer_idx % max_buffer_count,
        // xpbd_data->sa_async_iter_positions_cloth[buffer_idx %
        // max_buffer_count].size());
        return buffer_idx == Launcher::default_buffer_mask ? xpbd_data->sa_x : xpbd_data->sa_async_iter_positions_tet[buffer_idx % max_buffer_count];
    };
    auto fn_get_begin_buffer = [&](const uint buffer_idx) -> Buffer<Float3> & {
        // if constexpr (print_buffer_idx) fast_format("Begin buffer {} ({}) size =
        // {}", buffer_idx, buffer_idx % max_buffer_count,
        // xpbd_data->sa_async_begin_positions_cloth[buffer_idx %
        // max_buffer_count].size());
        return xpbd_data
            ->sa_async_begin_positions_tet[buffer_idx % max_buffer_count];
    };
    auto fn_copy_to_start_and_iter = [&](const Buffer<Float3> &input_array,
                                         const uint output_buffer_idx) {
        Buffer<Float3> &out1 = fn_get_begin_buffer(output_buffer_idx);
        Buffer<Float3> &out2 = fn_get_iter_buffer(output_buffer_idx);
        // if constexpr (print_buffer_idx) fast_format("fn_copy_to_start_and_iter
        // from {} to {}/{}", input_array.size(), out1.size(), out2.size());
        parallel_for(0, input_array.size(), [&](const uint vid) {
            Float3 input_vec = input_array[vid];
            out1[vid] = input_vec;
            out2[vid] = input_vec;
        });
    };
    auto fn_cloth_constraint_prev_func = [&](const Launcher::LaunchParam &param) {
        if constexpr (print_buffer_idx)
            fast_format("Prev get Buffer {}", param.buffer_idx);
        const float weight = 0.5f;

        if (!param.input_buffer_idxs.empty() &&
            param.left_buffer_idx != -1u)// Weight from left and input
        {
            for (const uint input_buffer_idx : param.input_buffer_idxs) {
                if constexpr (print_buffer_idx)
                    fast_format("Weight : from {} and {}", input_buffer_idx,
                                param.left_buffer_idx);
                auto &begin_buffer = param.is_allocated_to_main_device ? fn_get_begin_buffer(input_buffer_idx) : fn_get_begin_buffer(param.left_buffer_idx);
                parallel_for(0, mesh_data->num_verts_total, [&](const uint vid) {
                    Constrains::Core::read_and_solve_conflict(
                        vid, begin_buffer.data(), begin_buffer.data(),
                        fn_get_iter_buffer(input_buffer_idx).data(),
                        fn_get_iter_buffer(param.left_buffer_idx).data(), weight);
                });
            }
        } else if (!param.input_buffer_idxs.empty())// Copy from input
        {
            if constexpr (print_buffer_idx)
                fast_format("Copy input : from {} to {}",
                            param.input_buffer_idxs.back(), param.buffer_idx);
            fn_copy_to_start_and_iter(
                fn_get_iter_buffer(param.input_buffer_idxs.back()), param.buffer_idx);
        } else if (param.left_buffer_idx != -1u &&
                   param.left_buffer_idx !=
                       Launcher::input_buffer_mask)// Copy from left
        {
            // if constexpr (print_buffer_idx) fast_format("Copy left  : from {} to
            // {}", param.left_buffer_idx, param.buffer_idx);
            // fn_copy_to_start_and_iter(fn_get_iter_buffer(param.left_buffer_idx),
            // param.buffer_idx);
        } else if (param.left_buffer_idx ==
                   Launcher::input_buffer_mask)// Copy from sa_x
        {
            if constexpr (print_buffer_idx)
                fast_format("Copy predict position: from sa_x to {}", param.buffer_idx);
            fn_copy_to_start_and_iter(xpbd_data->sa_x, param.buffer_idx);
        }

        if (get_scene_params().print_xpbd_convergence && param.iter_idx == 0 &&
            param.cluster_idx == 0) {
            compute_energy(fn_get_iter_buffer(param.buffer_idx));
        }
    };
    auto fn_cloth_constraint_post_func = [&](const Launcher::LaunchParam &param) {
        if constexpr (print_buffer_idx)
            fast_format("Post get Buffer {}", param.buffer_idx);

        if (param.right_buffer_idx != -1u) {
            // Copying left operation should be done in the previous task, otherwise
            // we will get the value from the futher iterated buffer

            if constexpr (print_buffer_idx)
                fast_format("Copy right : from {} to {}", param.buffer_idx,
                            param.right_buffer_idx);
            fn_copy_to_start_and_iter(fn_get_iter_buffer(param.buffer_idx),
                                      param.right_buffer_idx);
        }
        // if (param.function_id == Launcher::id_vbd_all_in_one)
        // {
        //     fast_format("evaluate energy in cluster {} from buffer {} (iter_idx =
        //     {})", param.cluster_idx, param.buffer_idx, energy_idx);
        //     compute_energy(fn_get_iter_buffer(param.buffer_idx));
        //     return;
        // }

        if (get_scene_params().print_xpbd_convergence) {
            if (param.function_id ==
                    Launcher::
                        id_xpbd_constraint_self_collision_vv_half_cloth// Last task
                                                                       // of XPBD
                                                                       // (collision)
                ||
                param.function_id == Launcher::id_xpbd_constraint_last_node) {
                compute_energy(fn_get_iter_buffer(param.buffer_idx));
            }
        }
    };

    // Register Implementation

    // auto fn_launch = [&](const Launcher::LaunchParam& param) // Why cant i use
    // it in lambda ???
    {
        switch (param.function_id) {
            case Launcher::id_xpbd_predict_position: {
                predict_position();
                break;
            }
            case Launcher::id_xpbd_update_velocity: {
                update_velocity();
                break;
            }
            case Launcher::id_xpbd_reset_constrains: {
                reset_constrains();
                break;
            }
            case Launcher::id_xpbd_reset_collision_constrains: {
                reset_collision_constrains();
                break;
            }
            case Launcher::id_xpbd_constraint_last_node: {
                fn_cloth_constraint_prev_func(param);
                {}
                fn_cloth_constraint_post_func(param);
                parallel_copy(fn_get_iter_buffer(param.buffer_idx).data(),
                              xpbd_data->sa_x.data(), xpbd_data->sa_x.size());
                break;
            }
            case Launcher::id_xpbd_constraint_stress_half: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_tet_stress(fn_get_iter_buffer(param.buffer_idx), param.cluster_idx);
                fn_cloth_constraint_post_func(param);
                break;
            }
            case Launcher::id_xpbd_constraint_self_collision_vv_half_cloth: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_self_collision(fn_get_iter_buffer(param.buffer_idx), param.cluster_idx);
                fn_cloth_constraint_post_func(param);
                break;
            }
            case Launcher::id_xpbd_constraint_ground_collision_tet: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_ground_collision(fn_get_iter_buffer(param.buffer_idx));
                fn_cloth_constraint_post_func(param);
                break;
            }
            case Launcher::id_xpbd_constraint_obstacle_collision_vv_tet: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_obstacle_collision(fn_get_iter_buffer(param.buffer_idx));
                fn_cloth_constraint_post_func(param);
                break;
            }
            default: {
                fast_print_err("Illigal Input",
                               Launcher::taskNames.at(param.function_id));
                break;
            }
        }
    };
}
void GpuSolver::fn_dispatch(const Launcher::LaunchParam &param) {
    // Asynchronous iteration part
    constexpr uint max_buffer_count = 32;
    constexpr bool print_buffer_idx = false;
    auto fn_get_iter_buffer = [&](const uint buffer_idx) -> Buffer<Float3> & {
        return buffer_idx == Launcher::default_buffer_mask ? xpbd_data->sa_x : xpbd_data->sa_async_iter_positions_tet[buffer_idx % max_buffer_count];
    };
    auto fn_get_begin_buffer = [&](const uint buffer_idx) -> Buffer<Float3> & {
        return xpbd_data
            ->sa_async_begin_positions_tet[buffer_idx % max_buffer_count];
    };
    auto fn_copy_to_start_and_iter = [&](const Buffer<Float3> &input_array,
                                         const uint output_buffer_idx) {
        Buffer<Float3> &out1 = fn_get_begin_buffer(output_buffer_idx);
        Buffer<Float3> &out2 = fn_get_iter_buffer(output_buffer_idx);
        // if constexpr (print_buffer_idx) fast_format("fn_copy_to_start_and_iter
        // from {} to {}/{}", input_array.size(), out1.size(), out2.size());

        get_command_list().add_task(fn_copy_from_A_to_B_and_C);
        fn_copy_from_A_to_B_and_C.bind_ptr(input_array);
        fn_copy_from_A_to_B_and_C.bind_ptr(out1);
        fn_copy_from_A_to_B_and_C.bind_ptr(out2);
        fn_copy_from_A_to_B_and_C.launch_async(input_array.size());
    };
    auto fn_cloth_constraint_prev_func = [&](const Launcher::LaunchParam &param) {
        // if constexpr (print_buffer_idx) fast_format("Prev get Buffer {}",
        // param.buffer_idx);
        const float weight = 0.5f;

        if constexpr (print_buffer_idx)
            fast_format("    iter = {}, cluster = {}, input = {}, left = {}",
                        param.iter_idx, param.cluster_idx,
                        param.input_buffer_idxs.empty() ? "/" : std::to_string(param.input_buffer_idxs.back()),
                        param.left_buffer_idx == -1u ? "/" : std::to_string(param.left_buffer_idx));

        if (!param.input_buffer_idxs.empty() &&
            param.left_buffer_idx != -1u)// Weight from left and input
        {
            for (const uint input_buffer_idx : param.input_buffer_idxs) {
                if constexpr (print_buffer_idx)
                    fast_format("Weight : from {} and {}", input_buffer_idx,
                                param.left_buffer_idx);

                // Well we can always set GPU is the "main device"
                auto &begin_buffer = param.is_allocated_to_main_device ? fn_get_begin_buffer(input_buffer_idx) : fn_get_begin_buffer(param.left_buffer_idx);

                get_command_list().add_task(fn_read_and_solve_conflict);
                fn_read_and_solve_conflict.bind_ptr(begin_buffer);
                fn_read_and_solve_conflict.bind_ptr(begin_buffer);
                fn_read_and_solve_conflict.bind_ptr(
                    fn_get_iter_buffer(input_buffer_idx));
                fn_read_and_solve_conflict.bind_ptr(
                    fn_get_iter_buffer(param.left_buffer_idx));
                fn_read_and_solve_conflict.bind_constant(weight);
                fn_read_and_solve_conflict.launch_async(mesh_data->num_verts_total);
            }
        } else if (!param.input_buffer_idxs.empty())// Copy from input
        {
            if constexpr (print_buffer_idx)
                fast_format("Copy input : from {} to {}",
                            param.input_buffer_idxs.back(), param.buffer_idx);
            fn_copy_to_start_and_iter(
                fn_get_iter_buffer(param.input_buffer_idxs.back()), param.buffer_idx);
        } else if (param.left_buffer_idx != -1u &&
                   param.left_buffer_idx !=
                       Launcher::input_buffer_mask)// Copy from left
        {
            // Copying left operation should be done in the previous task, otherwise
            // we will get the value from the futher iterated buffer if constexpr
            // (print_buffer_idx) fast_format("Copy  left : from {} to {}",
            // param.left_buffer_idx, param.buffer_idx);
            // fn_copy_to_start_and_iter(fn_get_iter_buffer(param.left_buffer_idx),
            // param.buffer_idx);
        } else if (param.left_buffer_idx == Launcher::input_buffer_mask) {
            if constexpr (print_buffer_idx)
                fast_format("Copy predict position : from sa_x to {}",
                            param.buffer_idx);
            fn_copy_to_start_and_iter(xpbd_data->sa_x, param.buffer_idx);
        }

        // if (param.function_id == Launcher::id_vbd_all_in_one)
        // {
        //     // fast_format("bg evaluate energy in cluster {} from buffer {}
        //     (iter_idx = {})", param.cluster_idx, param.buffer_idx, energy_idx);
        //     compute_energy(fn_get_iter_buffer(param.buffer_idx));
        //     return;
        // }

        if (get_scene_params().print_xpbd_convergence && param.iter_idx == 0 &&
            param.cluster_idx == 0) {
            compute_energy(fn_get_iter_buffer(param.buffer_idx));
        }
    };
    auto fn_cloth_constraint_post_func = [&](const Launcher::LaunchParam &param) {
        // if constexpr (print_buffer_idx) fast_format("Post get Buffer {}",
        // param.buffer_idx);

        if (param.right_buffer_idx != -1u) {
            if constexpr (print_buffer_idx)
                fast_format("Copy right : from {} to {}", param.buffer_idx,
                            param.right_buffer_idx);
            fn_copy_to_start_and_iter(fn_get_iter_buffer(param.buffer_idx),
                                      param.right_buffer_idx);
        }

        // if (param.function_id == Launcher::id_vbd_all_in_one && (param.buffer_idx
        // == 3))
        // {
        //     fast_format("ed evaluate energy in cluster {} from buffer {}
        //     (iter_idx = {})", param.cluster_idx, param.buffer_idx, energy_idx);
        //     compute_energy(fn_get_iter_buffer(param.buffer_idx));
        //     return;
        // }

        if (get_scene_params().print_xpbd_convergence) {
            if (param.function_id ==
                    Launcher::
                        id_xpbd_constraint_self_collision_vv_half_cloth// Last task
                                                                       // of XPBD
                                                                       // (collision)
                || param.function_id == Launcher::id_xpbd_constraint_last_node) {
                compute_energy(fn_get_iter_buffer(param.buffer_idx));
            }
        }
    };

    // Register Implementation
    {
        switch (param.function_id) {
            case Launcher::id_xpbd_predict_position: {
                predict_position();
                break;
            }
            case Launcher::id_xpbd_update_velocity: {
                update_velocity();
                break;
            }
            case Launcher::id_xpbd_reset_constrains: {
                reset_constrains();
                break;
            }
            case Launcher::id_xpbd_reset_collision_constrains: {
                reset_collision_constrains();
                break;
            }
            case Launcher::id_xpbd_constraint_last_node: {
                fn_cloth_constraint_prev_func(param);
                {}
                {
                    fn_cloth_constraint_post_func(param);
                    get_command_list().add_task(fn_copy_from_A_to_B);
                    fn_copy_from_A_to_B.bind_ptr(fn_get_iter_buffer(param.buffer_idx));
                    fn_copy_from_A_to_B.bind_ptr(xpbd_data->sa_x);
                    fn_copy_from_A_to_B.launch_async(mesh_data->num_verts_total);
                }
                break;
            }
            case Launcher::id_xpbd_constraint_stress_half: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_tet_stress(fn_get_iter_buffer(param.buffer_idx), param.cluster_idx);
                fn_cloth_constraint_post_func(param);
                break;
            }
            case Launcher::id_xpbd_constraint_self_collision_vv_half_cloth: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_self_collision(fn_get_iter_buffer(param.buffer_idx), param.cluster_idx);
                fn_cloth_constraint_post_func(param);
                break;
            }
            case Launcher::id_xpbd_constraint_ground_collision_tet: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_ground_collision(fn_get_iter_buffer(param.buffer_idx));
                fn_cloth_constraint_post_func(param);
                break;
            }
            case Launcher::id_xpbd_constraint_obstacle_collision_vv_tet: {
                fn_cloth_constraint_prev_func(param);
                solve_constraint_obstacle_collision(fn_get_iter_buffer(param.buffer_idx));
                fn_cloth_constraint_post_func(param);
                break;
            }
            default: {
                fast_print_err("Illigal Input",
                               Launcher::taskNames.at(param.function_id));
                break;
            }
        }
    };
}
void GpuSolver::register_dag(Launcher::Scheduler &scheduler) {
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;
    {
        Launcher::Implementation ipm_xpbd_cpu(
            Launcher::DeviceTypeCpu, [&](const Launcher::LaunchParam &param) {
                cpu_solver->fn_dispatch(param);
            });
        Launcher::Implementation imp_xpbd_gpu(
            Launcher::DeviceTypeGpu,
            [&](const Launcher::LaunchParam &param) { this->fn_dispatch(param); });

        // Register DAG
        {
            std::vector<Launcher::Implementation>
                implementation_list_xpbd_cpu_and_gpu = {ipm_xpbd_cpu, imp_xpbd_gpu};

            // Init
            uint tid_xpbd_predict_position = scheduler.add_task(
                Launcher::Task(Launcher::id_xpbd_predict_position, 0,
                               implementation_list_xpbd_cpu_and_gpu));
            uint tid_xpbd_reset_constrains = scheduler.add_task(Launcher::Task(Launcher::id_xpbd_reset_constrains, 0, implementation_list_xpbd_cpu_and_gpu));
            uint tid_xpbd_reset_collision_constrains = scheduler.add_task(Launcher::Task(Launcher::id_xpbd_reset_collision_constrains, 0, implementation_list_xpbd_cpu_and_gpu));
            uint tid_xpbd_copy_current_position_to_2_devices = scheduler.add_task(Launcher::Task(Launcher::id_xpbd_copy_to_cpu_gpu, 0, implementation_list_xpbd_cpu_and_gpu));

            scheduler.set_connect(tid_xpbd_predict_position, tid_xpbd_copy_current_position_to_2_devices);

            // Solving Constraints
            std::vector<uint> prev_tids_stress;
            std::vector<uint> prev_tids_sdf_collision_vv_tet;
            std::vector<uint> prev_tids_self_collision_vv_tet;
            std::vector<uint> prev_prev_tids_self_collision_vv_tet;

            std::vector<std::vector<uint>> constraint_tasks;
            std::vector<uint> constraint_task_orders;
            const bool use_virtual_sync = true;
            const uint sync_frequece = 4;
            const uint sync_distance = 1;
            std::vector<uint> virtual_nodes;

            {
                xpbd_data->num_combined_clusters_stress = 8;
                xpbd_data->num_combined_clusters_self_collision = 1;
            }

            auto fn_connect_single_single = [&](const uint left, const uint right) {
                scheduler.set_connect(left, right);
            };
            auto fn_connect_single_multiple = [&](const uint left,
                                                  const std::vector<uint> &rights) {
                for (const uint &right : rights)
                    scheduler.set_connect(left, right);
            };
            auto fn_connect_multiple_single = [&](const std::vector<uint> &lefts,
                                                  const uint right) {
                for (const uint &left : lefts)
                    scheduler.set_connect(left, right);
            };
            auto fn_connect_multiple_multiple = [&](const std::vector<uint> &lefts,
                                                    const std::vector<uint> &rights) {
                for (const uint &left : lefts) {
                    for (const uint &right : rights) {
                        scheduler.set_connect(left, right);
                    }
                }
            };
            auto fn_add_brothers_to_graph = [&](
                                                const uint &iter_idx, const uint &constraint_idx,
                                                const Launcher::FunctionID &func_id,
                                                const uint &num_clusters) -> std::vector<uint> {
                std::vector<uint> tids_constraint(num_clusters);
                for (uint cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
                    const uint curr_idx = scheduler.add_task(Launcher::Task(func_id, iter_idx, constraint_idx, cluster_idx, implementation_list_xpbd_cpu_and_gpu));
                    tids_constraint[cluster_idx] = (curr_idx);
                }
                return tids_constraint;
            };

            std::vector<uint> sync_nodes;
            for (uint iter = 0; iter < constraint_iter_count; iter++) {

                const uint tid_xpbd_constraint_ground_collision_tet = scheduler.add_task(
                    Launcher::Task(Launcher::id_xpbd_constraint_ground_collision_tet, iter, CONSTRAINT_IDX_COLLISION, 0, implementation_list_xpbd_cpu_and_gpu));
                const uint tid_xpbd_constraint_obstacle_collision_tet = scheduler.add_task(
                    Launcher::Task(Launcher::id_xpbd_constraint_obstacle_collision_vv_tet, iter, CONSTRAINT_IDX_COLLISION, 0, implementation_list_xpbd_cpu_and_gpu));
                const std::vector<uint> tids_sdf_collision_vv_tet = {tid_xpbd_constraint_ground_collision_tet, tid_xpbd_constraint_obstacle_collision_tet};
                scheduler.set_connect(tid_xpbd_constraint_ground_collision_tet, tid_xpbd_constraint_obstacle_collision_tet);

                // Iteractive Constraints
                const std::vector<uint> tids_tet_stress = fn_add_brothers_to_graph(iter, CONSTRAINT_IDX_STRESS,
                                                                                   Launcher::id_xpbd_constraint_stress_half,
                                                                                   xpbd_data->num_combined_clusters_stress);

                const std::vector<uint> tids_self_collision_vv_tet = fn_add_brothers_to_graph(iter, CONSTRAINT_IDX_COLLISION,
                                                                                              Launcher::id_xpbd_constraint_self_collision_vv_half_tet,
                                                                                              xpbd_data->num_combined_clusters_self_collision);
                if (iter == 0) {
                    fn_connect_single_multiple(tid_xpbd_copy_current_position_to_2_devices, tids_tet_stress);
                    fn_connect_single_multiple(tid_xpbd_copy_current_position_to_2_devices, tids_sdf_collision_vv_tet);
                    fn_connect_single_multiple(tid_xpbd_copy_current_position_to_2_devices, tids_self_collision_vv_tet);
                    fn_connect_single_multiple(tid_xpbd_reset_constrains, tids_tet_stress);
                    fn_connect_single_multiple(tid_xpbd_reset_collision_constrains, tids_sdf_collision_vv_tet);
                    fn_connect_single_multiple(tid_xpbd_reset_collision_constrains, tids_self_collision_vv_tet);
                } else {
                    fn_connect_multiple_multiple(prev_tids_stress, tids_tet_stress);
                    fn_connect_multiple_multiple(prev_tids_sdf_collision_vv_tet, tids_sdf_collision_vv_tet);
                    fn_connect_multiple_multiple(prev_tids_self_collision_vv_tet, tids_self_collision_vv_tet);
                }

                // Set Virtual Connection
                {
                    std::vector<uint> curr_tasks;
                    curr_tasks.insert(curr_tasks.end(), tids_tet_stress.begin(), tids_tet_stress.end());
                    curr_tasks.insert(curr_tasks.end(), tids_sdf_collision_vv_tet.begin(), tids_sdf_collision_vv_tet.end());
                    curr_tasks.insert(curr_tasks.end(), tids_self_collision_vv_tet.begin(), tids_self_collision_vv_tet.end());

                    constraint_tasks.push_back(curr_tasks);
                    constraint_task_orders.insert(constraint_task_orders.end(), curr_tasks.begin(), curr_tasks.end());// fast_format("Insert In Iter {} = {}", iter, curr_tasks.size());
                    if (use_virtual_sync) {
                        const bool use_pipeline = true;
                        if (use_pipeline) {
                            auto half_stress = std::vector<uint>(tids_tet_stress.begin(), tids_tet_stress.begin() + tids_tet_stress.size() / 2);
                            fn_connect_multiple_multiple(half_stress, tids_sdf_collision_vv_tet);
                            fn_connect_multiple_multiple(tids_sdf_collision_vv_tet, tids_self_collision_vv_tet);
                        }
                    }
                }

                // Need To Set At Last : We Need The Previous Last Information To Connect With Constraints Above
                {
                    prev_tids_stress = tids_tet_stress;
                    prev_tids_sdf_collision_vv_tet = tids_sdf_collision_vv_tet;
                    prev_prev_tids_self_collision_vv_tet = prev_tids_self_collision_vv_tet;
                    prev_tids_self_collision_vv_tet = tids_self_collision_vv_tet;
                }
            }

            scheduler.set_constraint_task_orders(constraint_task_orders);

            // After All Iteration => Assemble And Update Velocity
            {
                uint last_node = scheduler.add_task(
                    Launcher::Task(Launcher::id_xpbd_constraint_last_node, 0,
                                   implementation_list_xpbd_cpu_and_gpu));
                uint tid_xpbd_update_velocity = scheduler.add_task(
                    Launcher::Task(Launcher::id_xpbd_update_velocity, 0,
                                   implementation_list_xpbd_cpu_and_gpu));

                if (!sync_nodes.empty()) scheduler.set_connect(sync_nodes.back(), last_node);
                fn_connect_multiple_single(prev_tids_stress, last_node);
                fn_connect_multiple_single(prev_tids_sdf_collision_vv_tet, last_node);
                fn_connect_multiple_single(prev_tids_self_collision_vv_tet, last_node);
                scheduler.set_connect(last_node, tid_xpbd_update_velocity);
            }
        }
    }
}
void GpuSolver::evaluate_compuatation_matrix(Launcher::Scheduler &scheduler) {
    // Init for computation matrix (Approximate value)

    std::vector<std::pair<Launcher::FunctionID, uint>> list_task_id = {};
    std::vector<std::vector<double>> list_cost;
    std::vector<double> cost_total;

    auto fn_reset_to_load = [&]() {
        parallel_for(0, mesh_data->num_verts_total, [&](uint vid) {
            Float3 saved_pos = mesh_data->sa_rest_position[vid];
            xpbd_data->sa_x_frame[vid] = saved_pos;

            Float3 saved_vel = mesh_data->sa_rest_velocity[vid];
            xpbd_data->sa_v_frame[vid] = saved_vel;
        });
    };
    auto func_prepare = []() {};

    const auto &list_task = scheduler.get_list_task();
    const auto &list_order = scheduler.get_list_order();
    const uint num_tasks = list_task.size();
    std::vector<std::vector<double>> cost_list_cpu(num_tasks);
    std::vector<std::vector<double>> cost_list_gpu(num_tasks);

    using CostMapKey = std::pair<Launcher::FunctionID, uint>;
    auto comp = [](const CostMapKey &key1, const CostMapKey &key2) {
        int func_id1 = int(key1.first);
        int func_id2 = int(key2.first);
        int cluster_id1 = int(key1.second);
        int cluster_id2 = int(key2.second);
        if (func_id1 != func_id2) {
            return func_id1 < func_id2;
        } else {
            return cluster_id1 < cluster_id2;
        }
    };

    struct CostMapCompFunc {
        using KeyType = std::pair<Launcher::FunctionID, uint>;
        bool operator()(const KeyType &key1, const KeyType &key2) const {
            int func_id1 = int(key1.first);
            int func_id2 = int(key2.first);
            int cluster_id1 = int(key1.second);
            int cluster_id2 = int(key2.second);
            if (func_id1 != func_id2)
                return func_id1 < func_id2;
            else
                return cluster_id1 < cluster_id2;
        }
    };
    using CostMapType = std::map<std::pair<Launcher::FunctionID, uint>,
                                 std::vector<double>, CostMapCompFunc>;
    CostMapType map_cpu;
    CostMapType map_gpu;

    const bool use_profiled_matrix = true;

    // Pre-Profiling
    if (!use_profiled_matrix) {
        const uint profile_cpu_loop_count = 50;
        const uint profile_gpu_loop_count = 50;
        const uint start_profile_threshhold = 20;

        auto fn_insert_cost_template =
            [](CostMapType &map, const Launcher::FunctionID &func_id,
               const uint &cluster_idx, const double &cost) -> void {
            const auto key = std::make_pair(func_id, cluster_idx);
            if (map.find(key) != map.end()) {
                map.at(key).push_back(cost);
            } else {
                map.insert(std::make_pair(key, std::vector<double>{cost}));
            }
        };
        auto fn_insert_cost_cpu = [&](const Launcher::FunctionID &func_id,
                                      const uint &cluster_idx, const double &cost) {
            fn_insert_cost_template(map_cpu, func_id, cluster_idx, cost);
        };
        auto fn_insert_cost_gpu = [&](const Launcher::FunctionID &func_id,
                                      const uint &cluster_idx, const double &cost) {
            fn_insert_cost_template(map_gpu, func_id, cluster_idx, cost);
        };

        auto fn_task_to_param = [](const Launcher::Task &task) {
            return Launcher::LaunchParam{
                .function_id = task.func_id,
                .iter_idx = task.iter_idx,
                .cluster_idx = task.cluster_idx,
                .is_allocated_to_main_device = true,
                .buffer_idx = Launcher::default_buffer_mask,
                .left_buffer_idx = -1u,
                .right_buffer_idx = -1u,
                .input_buffer_idxs = {},
            };
        };

        fn_reset_to_load();
        func_prepare();
        fn_reset_to_load();

        // Profile CPU
        fast_format("\nPrev CPU Loop for {} Times", 2);
        for (uint prev_loop = 0; prev_loop < 2; prev_loop++) {
            for (uint i = 0; i < 8; i++) {
                scheduler.launch(Launcher::Scheduler::LaunchModeCpu, fn_task_to_param,
                                 false);
            }
        }

        fast_print("CPU Loop...");
        double total_cpu = 0.0;
        for (uint loop = 0; loop < profile_cpu_loop_count; loop++) {
            // fast_print_single(loop); // We Do Not Do Print... Since It Is TOO SLOW
            // !!!
            SimClock clock_total;
            clock_total.start_clock();
            double sum_of_each_task = 0.0;
            for (auto tid : list_order) {
                SimClock clock;
                clock.start_clock();
                auto &task = list_task[tid];
                bool find;
                auto &imp = task.get_implementation(Launcher::DeviceTypeCpu, find);
                if (!find) {
                    fast_print_err("Does Not Exist CPU Implement");
                }
                { imp.launch_task(fn_task_to_param(task)); }
                double cost = clock.end_clock();
                if (loop > start_profile_threshhold) {
                    double dt = cost;
                    sum_of_each_task += dt;
                    cost_list_cpu[tid].push_back(dt);
                    fn_insert_cost_cpu(task.func_id, task.cluster_idx, dt);
                }
            }
            double curr_loop_cost = clock_total.end_clock();
            if (loop > start_profile_threshhold) {
                total_cpu += curr_loop_cost;
            }
        }
        cost_total.push_back(total_cpu / double(profile_cpu_loop_count -
                                                start_profile_threshhold - 1));

        fn_reset_to_load();

        // Profile GPU

        fast_print_single("GPU Loop...");
        double total_gpu = 0.0;
        auto &auto_fence_count = get_command_list().auto_fence_count;
        get_command_list().reset_auto_fence_count();
        for (uint loop = 0; loop < profile_gpu_loop_count; loop++) {
            fast_print_single(loop);
            get_command_list().reset_auto_fence_count();

            constexpr bool get_kernel_time = false;

            std::vector<double> list_gpu_costs;
            bool prev_is_gpu;
            std::vector<MTL::CommandBuffer *> curr_loop_buffers;
            for (uint i = 0; i < list_order.size(); i++) {
                auto tid = list_order[i % num_tasks];
                auto &task = list_task[tid];
                bool find;
                const auto &imp =
                    task.get_implementation(Launcher::DeviceTypeGpu, find);
                if (!find) {
                    if (prev_is_gpu) {
                        std::vector<double> prev_costs_from_cmd_buffer =
                            get_command_list().wait_all_cmd_buffers_and_get_costs(
                                get_kernel_time);
                        list_gpu_costs.insert(list_gpu_costs.end(),
                                              prev_costs_from_cmd_buffer.begin(),
                                              prev_costs_from_cmd_buffer.end());
                    }
                    SimClock clock_for_cpu;
                    clock_for_cpu.start_clock();
                    { imp.launch_task(fn_task_to_param(task)); }
                    list_gpu_costs.push_back(clock_for_cpu.end_clock());
                    if (loop == 0)
                        fast_print("Switch To CPU Implementation",
                                   Launcher::taskNames.at(task.func_id));
                    prev_is_gpu = false;
                } else {
                    auto buffer = get_command_list().start_new_list_with_new_buffer();
                    curr_loop_buffers.push_back(buffer);
                    { imp.launch_task(fn_task_to_param(task)); }
                    get_command_list()
                        .make_fence_with_previous_cmd_buffer();// If False, The Function
                                                               // May Be Empty
                    get_command_list().send_last_cmd_buffer_in_list();
                    prev_is_gpu = true;
                }
            }

            std::vector<double> rest_costs_from_buffer =
                get_command_list().wait_all_cmd_buffers_and_get_costs(
                    get_kernel_time);
            list_gpu_costs.insert(list_gpu_costs.end(),
                                  rest_costs_from_buffer.begin(),
                                  rest_costs_from_buffer.end());
            double duration = 1000.0 * (curr_loop_buffers.back()->GPUEndTime() -
                                        curr_loop_buffers[0]->GPUStartTime());
            double sum_of_each_task =
                std::accumulate(list_gpu_costs.begin(), list_gpu_costs.end(), 0.0);

            for (uint i = 0; i < list_order.size(); i++) {
                auto tid = list_order[i % num_tasks];
                const auto &task = list_task[tid];
                double dt = list_gpu_costs[i];

                if (loop != 0) {
                    cost_list_gpu[tid].push_back(dt);
                    fn_insert_cost_gpu(task.func_id, task.cluster_idx, dt);
                }
            }
            if (loop != 0)
                total_gpu += duration;
        }
        fast_print();
        cost_total.push_back(total_gpu / double(profile_gpu_loop_count - 1));

        fn_reset_to_load();

        const bool print_cost = true;
        {
            cost_total = {};

            if constexpr (print_cost)
                fast_print("Implementation List : ");
            if constexpr (print_cost) {
                std::cout << "{\n";
                for (const auto &pair : map_cpu) {
                    const auto &key = pair.first;
                    if (Launcher::taskNames.find(key.first) !=
                        Launcher::taskNames.end()) {
                        const std::string func_name = Launcher::taskNames.at(key.first);
                        std::cout << "        "
                                  << "    { Launcher::" << func_name << ", " << key.second
                                  << " }, \n";// std::format("{{}, {}}, ", func_name,
                                              // pair_cpu.second);
                    } else {
                        fast_print_err("Task does not exist");
                    }
                }
                std::cout << "}\n";
                fast_print("Cost List : ");
                std::cout << "{\n";
            }

            const uint drop_count = 2;
            for (const auto &pair : map_cpu) {
                const auto &key = pair.first;

                if constexpr (print_cost) {
                    if (key.second == 0) {
                        std::cout << "        "
                                  << "    // " << Launcher::taskNames.at(key.first) << "\n";
                    }
                }

                auto list_cpu = pair.second;
                auto list_gpu = map_gpu.at(key);
                std::sort(list_cpu.begin(), list_cpu.end());
                std::sort(list_gpu.begin(), list_gpu.end());

                // Drop the  largest 2 elements
                // Drop the smallest 2 elements
                double avg_cpu = std::accumulate(list_cpu.begin() + drop_count,
                                                 list_cpu.end() - drop_count, 0.0) /
                                 double(list_cpu.size() - 2 * drop_count);
                double avg_gpu = std::accumulate(list_gpu.begin() + drop_count,
                                                 list_gpu.end() - drop_count, 0.0) /
                                 double(list_gpu.size() - 2 * drop_count);

                list_task_id.push_back({key.first, key.second});
                list_cost.push_back({avg_cpu, avg_gpu});

                if constexpr (print_cost)
                    std::cout << "        "
                              << "    { " << avg_cpu << ", " << avg_gpu << " }, \n";
            }
            if constexpr (print_cost)
                std::cout << "}\n";

            // list_task_id.push_back({Launcher::id_additional_root, 0});
            // list_task_id.push_back({Launcher::id_additional_terminal, 0});
            // list_cost.push_back({0.0, 0.0});
            // list_cost.push_back({0.0, 0.0});
        }
    } else {
        // list_task_id = {
        //     { Launcher::id_xpbd_predict_position, 0 },
        //     { Launcher::id_xpbd_update_velocity, 0 },
        //     { Launcher::id_xpbd_constraint_last_node, 0 },
        // };
        // for (uint cluster = 0; cluster <
        // xpbd_data->num_clusters_per_vertex_bending; cluster++)
        // {
        //     list_task_id.push_back({Launcher::id_vbd_all_in_one, cluster});
        // }
        // for (uint i = 0; i < list_task_id.size(); i++)
        // {
        //     list_cost.push_back({1.0, 0.2});
        // }

        list_task_id = {
            {Launcher::id_xpbd_predict_position, 0},
            {Launcher::id_vbd_all_in_one, 0},
            {Launcher::id_vbd_all_in_one, 1},
            {Launcher::id_vbd_all_in_one, 2},
            {Launcher::id_vbd_all_in_one, 3},
            {Launcher::id_vbd_all_in_one, 4},
            {Launcher::id_vbd_all_in_one, 5},
            {Launcher::id_vbd_all_in_one, 6},
            {Launcher::id_vbd_all_in_one, 7},
            {Launcher::id_vbd_all_in_one, 8},
            {Launcher::id_vbd_all_in_one, 9},
            {Launcher::id_xpbd_constraint_last_node, 0},
            {Launcher::id_xpbd_update_velocity, 0},
            {Launcher::id_additional_root, 0},
            {Launcher::id_additional_terminal, 0},
        };
        list_cost = {
            // id_xpbd_constraint_last_node
            {0.00876, 0.00378333},
            // id_vbd_all_in_one
            {0.0911744, 0.0658269},
            {0.0962064, 0.0695517},
            {0.0955727, 0.0709349},
            {0.091657, 0.0711159},
            {0.0922093, 0.072056},
            {0.0870669, 0.0711635},
            {0.0816308, 0.0729224},
            {0.0662471, 0.0729321},
            {0.0276512, 0.0488068},
            {0.00111919, 0.0374237},
            // id_xpbd_predict_position
            {0.01712, 0.00437593},
            // id_xpbd_update_velocity
            {0.02156, 0.00476296},
        };
    }

    scheduler.profile_from(list_task_id, list_cost, cost_total);
}
void GpuSolver::physics_step_vbd_async() {
    xpbd_data->sa_x_step_start = xpbd_data->sa_x_frame;
    xpbd_data->sa_x = xpbd_data->sa_x_frame;
    xpbd_data->sa_v = xpbd_data->sa_v_frame;

    const uint num_substep = get_scene_params().print_xpbd_convergence ? 1 : get_scene_params().num_substep;
    const uint constraint_iter_count = get_scene_params().constraint_iter_count;

    xpbd_data->sa_system_energy.set_zero();
    energy_idx = 0;

    Launcher::Scheduler scheduler;
    scheduler.set_safety_check(false);

    SimClock scheule_clock;
    scheule_clock.start_clock();

    // Register DAG and implementation
    register_dag(scheduler);

    // Computation matrix can be updated per frame
    static std::vector<std::vector<float>>
        computation_matrix;// Update each frame, to fit the dynamic costs due to
                           // collisions

    // Set communication matrix
    {
        scheduler.communication_cost_matrix_uma = {
            {0.002, 0.220},/// gpu wait cpu
            {0.145, 0.01}  /// cpu wait gpu
        };
        scheduler.communication_speed_matrix = {};
        scheduler.communication_startup = {0, 0};// First call cost
    }

    // Make scheduling
    if (scheduler.topological_sort()) {
        if (computation_matrix.empty()) {
            evaluate_compuatation_matrix(scheduler);
        } else {
            scheduler.computation_matrix = computation_matrix;
        }

        scheduler.standardizing_dag({
            [&](const Launcher::LaunchParam &) {},
            [&](const Launcher::LaunchParam &) {
                get_command_list().add_task(fn_empty);
                fn_empty.launch_async(1);
            },
        });

        scheduler.scheduler_dag();

        scheduler.make_wait_events();
    }
    scheule_clock.end_clock();

    // LaunchModeFakeHetero
    // LaunchModeHetero
    // LaunchModeGpu
    // LaunchModePartialGPU
    // const auto launch_mode = Launcher::Scheduler::LaunchModeHetero;
    const auto launch_mode =
        (Launcher::Scheduler::LaunchMode)get_scene_params().launch_mode;

    // Run
    SimClock clock;
    clock.start_clock();
    float frame_cost = 0.0f;
    for (uint substep = 0; substep < num_substep; substep++) {
        {
            get_scene_params().current_substep = substep;
        }
        SimClock substep_clock;
        substep_clock.start_clock();

        // In this mode, you will run scheduled tasks with SYNC waiting
        // The final result should be the same as LaunchModeHetero
        // (Since we use multi-buffer to identity the inputs, so if we miss the
        // relationship, we will get NAN or exposition) We will use runtime
        // profiling to update the computation matrix and re-schedule
        if (launch_mode == Launcher::Scheduler::LaunchModeFakeHetero) {
            auto fn_task_to_param = [](const Launcher::Task &task) {
                // task.print_with_cluster(0);
                return Launcher::LaunchParam{
                    .function_id = task.func_id,
                    .iter_idx = task.iter_idx,
                    .cluster_idx = task.cluster_idx,
                    .is_allocated_to_main_device = task.is_allocated_to_main_device,
                    .buffer_idx = task.buffer_idx,
                    .left_buffer_idx = task.buffer_left,
                    .right_buffer_idx = task.buffer_right,
                    .input_buffer_idxs = task.buffer_ins,
                };
            };
            scheduler.launch(Launcher::Scheduler::LaunchModeFakeHetero,
                             fn_task_to_param, false);
        }

        // In this mode, you will run scheduled tasks with ASYNC waiting, the actual
        // time should close to the scheduling time (after seceral frames) However,
        // this mode not work (e.g., GPU being locked or the simulation result is
        // not equal to 'LaunchModeFakeHetero')
        //                              when there are too many tasks (e.g. 40
        //                              command-buffers on the GPU) (Another reason
        //                              that the simulation result is not equal to
        //                              'LaunchModeFakeHetero' is that
        //                                 we may not write the data transfering
        //                                 stragegy correctly which may result
        //                                 buffer conflict or access)
        // This is limited to the hardware, maybe we can solve it by segmenting the
        // commission of gpu commands If you have some ideas to fix it, hope you can
        // help me (you find my contact information in my homepage:
        // https://chengzhuuwu.github.io/)
        else if (launch_mode == Launcher::Scheduler::LaunchModeHetero ||
                 launch_mode == Launcher::Scheduler::LaunchModePartialGPU) {
            auto fn_task_to_param = [](const Launcher::Task &task) {
                // task.print_with_cluster(0);
                return Launcher::LaunchParam{
                    .function_id = task.func_id,
                    .iter_idx = task.iter_idx,
                    .cluster_idx = task.cluster_idx,
                    .is_allocated_to_main_device = task.is_allocated_to_main_device,
                    .buffer_idx = task.buffer_idx,
                    .left_buffer_idx = task.buffer_left,
                    .right_buffer_idx = task.buffer_right,
                    .input_buffer_idxs = task.buffer_ins,
                };
            };
            scheduler.launch(launch_mode, fn_task_to_param, false);
            // scheduler.launch(Launcher::Scheduler::LaunchModeHetero,
            // fn_task_to_param, false);
            // scheduler.launch(Launcher::Scheduler::LaunchModePartialGPU,
            // fn_task_to_param, false);
        }

        // In this mode, you will run tasks sorted by ranku on single device
        else if (launch_mode == Launcher::Scheduler::LaunchModeCpu ||
                 launch_mode == Launcher::Scheduler::LaunchModeGpu) {
            auto fn_task_to_param = [](const Launcher::Task &task) {
                return Launcher::LaunchParam{
                    .function_id = task.func_id,
                    .iter_idx = task.iter_idx,
                    .cluster_idx = task.cluster_idx,
                    .is_allocated_to_main_device = true,
                    .buffer_idx = Launcher::default_buffer_mask,
                    .left_buffer_idx =
                        -1u,// We do not use asynchronous iteration on sequential mode
                    .right_buffer_idx =
                        -1u,// Sin it brings more cost on data copying and weighting
                    .input_buffer_idxs = {},
                };
            };
            scheduler.launch(launch_mode, fn_task_to_param, false);
            // scheduler.launch(Launcher::Scheduler::LaunchModeCpu, fn_task_to_param,
            // false);
        }

        // substep_clock.end_clock(); frame_cost += substep_clock.duration();
    }
    // frame_cost /= num_substep;
    frame_cost = clock.end_clock();

    computation_matrix = scheduler.computation_matrix;
    scheduler.update_costs_from_computation_matrix();

    if (launch_mode == Launcher::Scheduler::LaunchModeHetero) {
        fast_format("   In Frame {:2} : Hybrid Cost/Desire = {:.2f}/{:5.2f}, "
                    "speedup = {:5.2f}%/{:5.2f}% to GPU/CPU (profile time = "
                    "{:5.2f}/{:5.2f}), scheuling cost = {:3.2f}",
                    get_scene_params().current_frame, frame_cost,
                    scheduler.get_scheduled_end_time() *
                        num_substep,// get_scheduled_end_time() should near to
                                    // actual time in LaunchModeHetero
                    scheduler.get_scheduled_speedups()[1] * 100,
                    scheduler.get_scheduled_speedups()[0] * 100,
                    num_substep * scheduler.get_proc_costs()[1],// GPU is proc 1
                    num_substep * scheduler.get_proc_costs()[0] // CPU is proc 0
                    ,
                    scheule_clock.duration());
    } else {
        fast_format(
            "   In Frame {} : {} Cost = {:6.3f}", get_scene_params().current_frame,
            launch_mode == Launcher::Scheduler::LaunchModeCpu ? "CPU" : "GPU",
            frame_cost);
    }

    if (get_scene_params().current_frame == 29 &&
        launch_mode == Launcher::Scheduler::LaunchModeHetero)
        scheduler.print_task_costs_map();

    {
        if (get_scene_params().print_xpbd_convergence) {
            std::vector<double> list_energy(energy_idx);
            for (uint it = 0; it < list_energy.size(); it++) {
                list_energy[it] = xpbd_data->sa_system_energy[it];
            }
            fast_print_iterator(list_energy, "Energy Convergence");
            energy_idx = 0;
        }
    }

    xpbd_data->sa_x_frame = xpbd_data->sa_x;
    xpbd_data->sa_v_frame = xpbd_data->sa_v;
}
void CpuSolver::solve_constraints_XPBD() {
    auto &iter_position = xpbd_data->sa_x;

    if (get_scene_params().print_xpbd_convergence &&
        get_scene_params().current_it == 0) {
        compute_energy(iter_position);
    }

    {
        for (uint i = 0; i < xpbd_data->num_clusters_tet_stress; i++) {
            solve_constraint_tet_stress(iter_position, i);
        }
        solve_constraint_ground_collision(iter_position);
        solve_constraint_obstacle_collision(iter_position);
        for (uint i = 0; i < xpbd_data->num_combined_clusters_self_collision; i++) {
            solve_constraint_self_collision(iter_position, i);
        }
    }

    if (get_scene_params().print_xpbd_convergence) {
        compute_energy(iter_position);
    }
}
void GpuSolver::solve_constraints_XPBD() {
    auto &iter_position = xpbd_data->sa_x;

    if (get_scene_params().print_xpbd_convergence &&
        get_scene_params().current_it == 0) {
        compute_energy(iter_position);
    }

    {
        for (uint i = 0; i < xpbd_data->num_clusters_tet_stress; i++) {
            solve_constraint_tet_stress(iter_position, i);
        }
        solve_constraint_ground_collision(iter_position);
        solve_constraint_obstacle_collision(iter_position);
        for (uint i = 0; i < xpbd_data->num_combined_clusters_self_collision; i++) {
            solve_constraint_self_collision(iter_position, i);
        }
    }

    if (get_scene_params().print_xpbd_convergence) {
        compute_energy(iter_position);
    }
}

enum class SolverType {
    GaussNewton,
    XPBD_CPU,
    XPBD_GPU,
    XPBD_async,
};

class SolverInterface {
public:
    SolverInterface() {}
    ~SolverInterface() {}

    void set_data_pointer(
        XpbdData *xpbd_data,
        TetData *mesh_data,
        ObstacleData *obstacle_data,
        VivaceColoringData *coloring_data) {
        this->xpbd_data = xpbd_data;
        this->mesh_data = mesh_data;
        this->obstacle_data = obstacle_data;
        this->coloring_data = coloring_data;

        this->lbvh_data_obstacle = &xpbd_data->lbvh_data_obstacle;
        this->lbvh_data_tet = &xpbd_data->lbvh_data_tet;
        this->self_collision_data_tet = &xpbd_data->tet_collision;
        this->obstacle_collision_data_tet = &xpbd_data->obs_collision_tet;
    }
    void set_solver_pointer(
        LbvhFaceEdge<LBVHUpdateTypeObstacle> *lbvh_obstacle,
        LbvhFaceEdge<LBVHUpdateTypeCloth> *lbvh_tet,
        RandomGraphColoringCPU *vivace_cpu,
        RandomGraphColoringGPU *vivace_gpu,
        CpuSolver *cpu_solver,
        GpuSolver *gpu_solver) {
        this->lbvh_obstacle = lbvh_obstacle;
        this->lbvh_tet = lbvh_tet;
        this->vivace_cpu = vivace_cpu;
        this->vivace_gpu = vivace_gpu;
        this->cpu_solver = cpu_solver;
        this->gpu_solver = gpu_solver;
    }
    void register_solver_type(SolverType type) {
        if (type == SolverType::GaussNewton) {
            fast_format_err("Empty NewtonSolver implementation");
        } else {
            cpu_solver->get_data_pointer(xpbd_data, mesh_data, obstacle_data, coloring_data,
                                         lbvh_data_obstacle, lbvh_data_tet,
                                         self_collision_data_tet,
                                         obstacle_collision_data_tet);
            gpu_solver->get_data_pointer(xpbd_data, mesh_data, obstacle_data, coloring_data,
                                         lbvh_data_obstacle, lbvh_data_tet,
                                         self_collision_data_tet,
                                         obstacle_collision_data_tet);
            cpu_solver->init_xpbd_system();
            gpu_solver->init_xpbd_system();

            CpuSolver::init_simulation_params();
        }
    }

public:
    void physics_step(SolverType type);
    void restart_system();
    void save_mesh_to_obj(const std::string &addition_str = "");

private:
    XpbdData *xpbd_data;
    TetData *mesh_data;
    ObstacleData *obstacle_data;
    VivaceColoringData *coloring_data;
    LbvhFaceEdgeData *lbvh_data_obstacle;
    LbvhFaceEdgeData *lbvh_data_tet;
    XpbdSelfCollision *self_collision_data_tet;
    XpbdObstacleCollision *obstacle_collision_data_tet;
    VivaceColoringData *coloring_data_tet;

    LbvhFaceEdge<LBVHUpdateTypeObstacle> *lbvh_obstacle;
    LbvhFaceEdge<LBVHUpdateTypeCloth> *lbvh_tet;
    RandomGraphColoringCPU *vivace_cpu;
    RandomGraphColoringGPU *vivace_gpu;

private:
    CpuSolver *cpu_solver;
    GpuSolver *gpu_solver;
};

void SolverInterface::restart_system() {
    parallel_for(0, mesh_data->num_verts_total, [&](uint vid) {
        Float3 rest_pos = mesh_data->sa_rest_position[vid];
        xpbd_data->sa_x_frame[vid] = rest_pos;

        Float3 rest_vel = mesh_data->sa_rest_velocity[vid];
        xpbd_data->sa_v[vid] = rest_vel;
    });
}
void SolverInterface::physics_step(SolverType type) {
    switch (type) {
        case SolverType::XPBD_CPU: {
            cpu_solver->physics_step_xpbd();/////////////
            break;
        }
        case SolverType::XPBD_GPU: {
            gpu_solver->physics_step_xpbd();/////////////
            break;
        }
        case SolverType::XPBD_async: {
            gpu_solver->physics_step_vbd_async();/////////////
            break;
        }
        default: {
            fast_format_err("Emptey solver");
            break;
        }
    }

    {
        // Other operations...
    }
}
void SolverInterface::save_mesh_to_obj(const std::string &addition_str) {
    const std::string filename = std::format(
        "frame_{}{}.obj", get_scene_params().current_frame, addition_str);

    std::string full_directory =
        std::string(SELF_RESOURCES_PATH) + std::string("/outputs/");

    {
        std::filesystem::path dir_path(full_directory);
        if (!std::filesystem::exists(dir_path)) {
            try {
                std::filesystem::create_directories(dir_path);
                std::cout << "Created directory: " << dir_path << std::endl;
            } catch (const std::filesystem::filesystem_error &e) {
                std::cerr << "Error creating directory: " << e.what() << std::endl;
                return;
            }
        }
    }

    std::string full_path = full_directory + filename;
    std::ofstream file(full_path, std::ios::out);

    if (file.is_open()) {
        file << "# File Simulated From SIGGRAPH 2025 paper <Automatic Task "
                "Scheduling for Cloth and Deformable Simulation on Heterogeneous "
                "Environments>"
             << std::endl;

        uint glocal_vert_id_prefix = 0;
        uint glocal_mesh_id_prefix = 0;

        // Cloth Part
        // if (get_scene_params().draw_cloth)
        {
            // for (uint clothIdx = 0; clothIdx < cloth_data.num_cloths; clothIdx++)
            const uint clothIdx = 0;
            {
                file << "o mesh_" << (glocal_mesh_id_prefix + clothIdx) << std::endl;
                for (uint vid = 0; vid < mesh_data->num_verts_total; vid++) {
                    const Float3 vertex = xpbd_data->sa_x_frame[vid];
                    file << "v " << vertex.x << " " << vertex.y << " " << vertex.z
                         << std::endl;
                }

                for (uint fid = 0; fid < mesh_data->num_surface_faces_total; fid++) {
                    const Int3 f = mesh_data->sa_surface_faces[fid] + makeInt3(1) +
                                   makeInt3(glocal_vert_id_prefix);
                    file << "f " << f.x << " " << f.y << " " << f.z << std::endl;
                }
            }
            glocal_vert_id_prefix += mesh_data->num_verts_total;
            glocal_mesh_id_prefix += 1;
        }

        file.close();
        std::cout << "OBJ file saved: " << full_path << std::endl;
    } else {
        std::cerr << "Unable to open file: " << full_path << std::endl;
    }
}

int main() {
    std::cout << "Hello, Asynchronous Iteration!" << std::endl;

    // Init metal system
    {
        create_device();

        init_command_list();

        init_scene_params();
    }

    // Init mesh
    TetData mesh_data;
    { init_tet_mesh(&mesh_data); }

    ObstacleData obstacle_data;
    { init_obstacle_mesh(&obstacle_data); }

    XpbdData xpbd_data;
    { init_xpbd_data(&mesh_data, &obstacle_data, &xpbd_data); }

    VivaceColoringData coloring_data;
    { coloring_data.resize(mesh_data.num_verts_total); }

    LbvhFaceEdge<LBVHUpdateTypeObstacle> lbvh_obstacle;
    LbvhFaceEdge<LBVHUpdateTypeCloth> lbvh_tet;
    RandomGraphColoringCPU vivace_cpu;
    RandomGraphColoringGPU vivace_gpu;
    CpuSolver cpu_solver;
    GpuSolver gpu_solver;
    {
        lbvh_tet.vert_cpu.init_cloth_lbvh(xpbd_data.lbvh_data_tet.vert_tree);
        lbvh_tet.face_cpu.init_cloth_lbvh(xpbd_data.lbvh_data_tet.face_tree);
        lbvh_tet.edge_cpu.init_cloth_lbvh(xpbd_data.lbvh_data_tet.edge_tree);
        lbvh_obstacle.vert_cpu.init_cloth_lbvh(xpbd_data.lbvh_data_obstacle.vert_tree);
        lbvh_obstacle.face_cpu.init_cloth_lbvh(xpbd_data.lbvh_data_obstacle.face_tree);
        lbvh_obstacle.edge_cpu.init_cloth_lbvh(xpbd_data.lbvh_data_obstacle.edge_tree);

        vivace_cpu.init_graph_coloring_system(coloring_data, xpbd_data.tet_collision);
        vivace_gpu.init_graph_coloring_system(coloring_data, xpbd_data.tet_collision, vivace_cpu);
    }

    // Init solver
    SolverInterface solver;
    {
        solver.set_data_pointer(&xpbd_data, &mesh_data, &obstacle_data, &coloring_data);

        solver.set_solver_pointer(&lbvh_obstacle, &lbvh_tet, &vivace_cpu, &vivace_gpu, &cpu_solver, &gpu_solver);

        solver.register_solver_type(SolverType::XPBD_CPU);
    }

    // Some params
    {
        get_scene_params().use_substep = false;
        get_scene_params().num_substep = 10;
        get_scene_params().constraint_iter_count =
            12;// May not be too large, otherwise communcation overload on GPU may
               // be higher
        get_scene_params().use_bending = true;
        get_scene_params().use_quadratic_bending_model = true;
        get_scene_params().print_xpbd_convergence = false;
        get_scene_params().use_xpbd_solver = false;
        get_scene_params().use_vbd_solver = true;
    }

    const uint max_frame = 30;

    // Synchronous CPU Implementation
    {
        solver.restart_system();
        solver.save_mesh_to_obj("_init");
        fast_format("");
        fast_format("");
        fast_format("Sync CPU part");
        get_scene_params().launch_mode = Launcher::Scheduler::LaunchModeCpu;
    }
    {
        for (uint frame = 0; frame < max_frame; frame++) {
            get_scene_params().current_frame = frame;

            solver.physics_step(SolverType::XPBD_CPU);
        }
    }
    {
        // solver.save_mesh_to_obj("_sync_CPU");
    }

    return 0;

    // Synchronous GPU Implementation
    {
        solver.restart_system();
        fast_format("Sync GPU part");
        get_scene_params().launch_mode = Launcher::Scheduler::LaunchModeGpu;
    }
    {
        for (uint frame = 0; frame < max_frame; frame++) {
            get_scene_params().current_frame = frame;

            // solver.physics_step(SolverTypeVBD_GPU);
            solver.physics_step(SolverType::XPBD_GPU);
        }
    }
    {
        // solver.save_mesh_to_obj("_sync_GPU");
    }

    // Asynchronous Implementation
    {
        solver.restart_system();
        fast_format("Hybrid part");
        get_scene_params().launch_mode = Launcher::Scheduler::LaunchModeHetero;
    }
    {
        for (uint frame = 0; frame < max_frame; frame++) {
            get_scene_params().current_frame = frame;

            solver.physics_step(SolverType::XPBD_async);
        }
    }
    { solver.save_mesh_to_obj("_hybrid_CPU_GPU"); }

    return 0;
}