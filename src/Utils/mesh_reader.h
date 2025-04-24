#pragma once

#include <string>
#include <tiny_obj_loader.h>
#include "TargetConfig.h"
#include "aabb.h"
#include "shared_array.h"
#include <iostream>
#include <string>
#include <fstream>

// typedef OpenMesh::TriMesh_ArrayKernelT<>  MeshInfo;
using MeshShape = std::vector<tinyobj::shape_t>;
using MeshAttrib= tinyobj::attrib_t;
using MeshMat = std::vector<tinyobj::material_t>;

struct TriangleMeshData 
{
    std::vector<Float3> model_positions;
    std::vector<Float2> uv_positions;
    std::vector<Int3> faces;
    std::vector<Int3> normal_faces;
    std::vector<Int3> texcoord_faces;
    std::vector<Int3> invalid_faces;
    std::vector<Int3> invalid_normal_faces;
    std::vector<Int3> invalid_texcoord_faces;
    std::vector<uint> uv_to_vert_map;
    std::vector<Int2> edges;
    std::vector<Int4> bending_edges;
    std::vector<Int2> boundary_edges;
    std::vector<Int2> edge_adj_faces;
    std::vector<Int2> bending_edge_adj_faces;

    std::vector<std::string> material_names;
    std::vector<int> material_ids;
    std::vector<int> invalid_material_ids;

    bool has_uv = false;
    
    TriangleMeshData() {}
    TriangleMeshData(const TriangleMeshData& input) :
        model_positions(input.model_positions),
        uv_positions(input.uv_positions),
        faces(input.faces),
        normal_faces(input.normal_faces),
        texcoord_faces(input.texcoord_faces),
        invalid_faces(input.invalid_faces),
        invalid_normal_faces(input.invalid_normal_faces),
        invalid_texcoord_faces(input.invalid_texcoord_faces),
        uv_to_vert_map(input.uv_to_vert_map),
        edges(input.edges),
        bending_edges(input.bending_edges),
        has_uv(input.has_uv),
        boundary_edges(input.boundary_edges),
        edge_adj_faces(input.edge_adj_faces),
        bending_edge_adj_faces(input.bending_edge_adj_faces),
        material_names(input.material_names) ,
        material_ids(input.material_ids),
        invalid_material_ids(input.invalid_material_ids) 
        {}
};
struct TetrahedralMeshData 
{
    std::vector<Float3> model_positions;
    // std::vector<Float2> surface_uv_positions;
    std::vector<Int4> tetrahedral;
    std::vector<Int3> surface_faces;
    std::vector<Int2> surface_edges;
    std::vector<uint> surface_verts;
};

namespace SimMesh{

inline void extract_surface_face_and_vert_from_tets(
    const std::vector<Float3>& input_position,
    const std::vector<Int4>& input_tets,
    std::vector<uint>& inner_tets, std::vector<uint>& outer_tets, std::vector<Int3>& surface_faces, std::vector<uint>& surface_verts)
{
    const uint num_tets = input_tets.size();
    const uint num_verts = input_position.size();
    
    std::vector<bool> list_vert_is_on_surface(num_verts, false);
    std::vector<Int4> tmp_tets(num_tets * 4);
    
    auto tet_local_sort = [](Int4 vids) -> Int4
    {
        uint tmp[4] = {vids[0], vids[1], vids[2], vids[3]};
        std::sort(tmp, tmp + 4);
        return makeInt4(tmp[0], tmp[1], tmp[2], tmp[3]);
    };
    parallel_for(0, num_tets, [&](uint tid)
    {
        Int4 tet = input_tets[tid];
        tet = tet_local_sort(tet);
        tmp_tets[4 * tid + 0] = makeInt4(tet[0], tet[1], tet[2], tid);
        tmp_tets[4 * tid + 1] = makeInt4(tet[0], tet[1], tet[3], tid);
        tmp_tets[4 * tid + 2] = makeInt4(tet[0], tet[2], tet[3], tid);
        tmp_tets[4 * tid + 3] = makeInt4(tet[1], tet[2], tet[3], tid);
    });
    std::sort(tmp_tets.begin(), tmp_tets.end(), [](const Int4& left, const Int4& right)
    {
        int temp;
        temp = left[0] - right[0]; if(temp != 0) return temp < 0;
        temp = left[1] - right[1]; if(temp != 0) return temp < 0;
        temp = left[2] - right[2]; if(temp != 0) return temp < 0;
        temp = left[3] - right[3];               return temp < 0;
    });
    std::vector<uchar> list_face_type(tmp_tets.size(), 0);
    parallel_for(0, tmp_tets.size(), [&](const uint i)
    {
        Int4 curr_face = tmp_tets[i];
        if (i != tmp_tets.size() - 1) { Int4 next_face = tmp_tets[i + 1]; if (next_face[0] == curr_face[0] && next_face[1] == curr_face[1] && next_face[2] == curr_face[2]) list_face_type[i] = 1; }
        if (i != 0)                   { Int4 prev_face = tmp_tets[i - 1]; if (prev_face[0] == curr_face[0] && prev_face[1] == curr_face[1] && prev_face[2] == curr_face[2]) list_face_type[i] = 2; }
    });
    
    uint num_surface_faces = 0;
    for (const auto& value : list_face_type) { if (value == 0) num_surface_faces++; }
    surface_faces.resize(num_surface_faces);

    auto make_ordered_face = [&input_position](const Int3& unorderd_face, const Int4& orig_tet) -> Int3 
    {
        const uint v1 = unorderd_face[0];
        const uint v2 = unorderd_face[1];
        const uint v3 = unorderd_face[2];
        const uint opposite_vertex = 
            (orig_tet[0] + orig_tet[1]+ orig_tet[2] + orig_tet[3]) - 
            (unorderd_face[0] + unorderd_face[1] + unorderd_face[2]);
        Float3 vec1 = input_position[v2] - input_position[v1];
        Float3 vec2 = input_position[v3] - input_position[v1];
        Float3 normal = cross_vec(vec1, vec2);
        Float3 vec_to_opposite = input_position[opposite_vertex] - input_position[v1];
        
        if (dot_vec(normal, vec_to_opposite) > 0) 
            return makeInt3(v1, v3, v2);  // Swap v2 and v3 to reverse the order
        else
            return makeInt3(v1, v2, v3);  // Correct order
    };


    parallel_for_and_scan(0, tmp_tets.size(), [&](const uint i)
    {
        const auto face_type = list_face_type[i];
        if (face_type == 0)  // Boundary Face
        {
            return 1;
        }
        else if (face_type == 1) // Inner Faces
        {
            return 0;
        }
        return 0;    
    }, 
    [&](const uint i, const uint& prefix, const uint& curr_return)
    {
        if (curr_return == 1)  // Boundary Face
        {
            const uint fid = prefix - 1;
            const Int4 curr_value = tmp_tets[i];
            const Int3 face = makeInt3(curr_value[0], curr_value[1], curr_value[2]);
            const uint tetIdx = curr_value[3];

            Int4 orig_tet = input_tets[tetIdx];
            surface_faces[fid] = make_ordered_face(face, orig_tet);
            list_vert_is_on_surface[curr_value[0]] = true;
            list_vert_is_on_surface[curr_value[1]] = true;
            list_vert_is_on_surface[curr_value[2]] = true;
        }
    }, 
    0u);

    std::vector<bool> is_surface_vert(num_verts, false);

    uint num_surface_verts = parallel_for_and_reduce_sum<uint>(0, num_verts, [&](const uint vid) { return list_vert_is_on_surface[vid] ? 1 : 0; });
    surface_verts.resize(num_surface_verts);
    parallel_for_and_scan(0, num_verts, [&](const uint vid)
    {
        return list_vert_is_on_surface[vid] ? 1 : 0;
    }, 
    [&](const uint vid, const uint& prefix, const uint& curr_return)
    {
        if (curr_return == 1)
        {
            surface_verts[prefix - 1] = vid;
            is_surface_vert[vid] = true;
        }
    }, 0u);

    single_thread_for(0, num_tets, [&](uint tid)
    {
        Int4 tet = input_tets[tid];
        if (   is_surface_vert[tet[0]]
            || is_surface_vert[tet[1]]
            || is_surface_vert[tet[2]]
            || is_surface_vert[tet[3]])
        {
            outer_tets.push_back(tid);
        }
        else 
        {
            inner_tets.push_back(tid);
        }
    });

}

template<bool extract_bending_edge>
inline void extract_edges_from_surface(
    const std::vector<Int3>& input_faces,
    std::vector<Int2>& output_edges, std::vector<Int4>& output_bending_edges)
{
    const uint num_surface_faces = input_faces.size();
    std::vector<Int3> tmp_faces(num_surface_faces * 3);

    auto face_local_sort = [](Int3 vids) -> Int3
    {
        uint tmp[3] = {vids[0], vids[1], vids[2]};
        std::sort(tmp, tmp + 3);
        return makeInt3(tmp[0], tmp[1], tmp[2]);
    };
    parallel_for(0, num_surface_faces, [&](const uint fid)
    {
        Int3 face = input_faces[fid];
        face = face_local_sort(face);
        tmp_faces[3 * fid + 0] = makeInt3(face[0], face[1], fid);
        tmp_faces[3 * fid + 1] = makeInt3(face[0], face[2], fid);
        tmp_faces[3 * fid + 2] = makeInt3(face[1], face[2], fid);
    });
    parallel_sort(tmp_faces.begin(), tmp_faces.end(), [](const Int3& left, const Int3& right)
    {
        int temp;
        temp = left[0] - right[0]; if(temp != 0) return temp < 0;
        temp = left[1] - right[1]; if(temp != 0) return temp < 0;
        temp = left[2] - right[2];               return temp < 0;
    });
    std::vector<uchar> list_edge_type(tmp_faces.size(), 0);
    parallel_for(0, tmp_faces.size(), [&](const uint i)
    {
        Int3 curr_face = tmp_faces[i];
        if (i != tmp_faces.size() - 1) { Int3 next_face = tmp_faces[i + 1]; if (next_face[0] == curr_face[0] && next_face[1] == curr_face[1]) list_edge_type[i] = 1; }
        if (i != 0)                    { Int3 prev_face = tmp_faces[i - 1]; if (prev_face[0] == curr_face[0] && prev_face[1] == curr_face[1]) list_edge_type[i] = 2; }
    });

    uint num_edges = 0; uint num_bending_edges = 0;
    for (const auto& value : list_edge_type) { if (value == 0 || value == 2) num_edges++; if (value == 1) num_bending_edges++; }
    output_edges.resize(num_edges);
    output_bending_edges.resize(num_bending_edges);

    parallel_for_and_scan(0, tmp_faces.size(), [&](const uint i)
    {
        // edge_type:
        //      0 : Boundary
        //      1 : Inner edges (left)  (Same As Its Right)
        //      2 : Inner edges (right) (Same As Its Left)
        auto edge_type = list_edge_type[i];
        if (edge_type == 0 || edge_type == 2) // Inner Edges
            return 1;
        else 
            return 0;
    }, 
    [&](const uint i, const uint& prefix, const uint& curr_return)
    {
        if (curr_return == 1)
        {
            const uint eid = prefix - 1;
            const Int3 curr_value = tmp_faces[i];
            output_edges[eid] = makeInt2(curr_value[0], curr_value[1]);
        }
    }, 0u);

    if (extract_bending_edge)
    {
        parallel_for_and_scan(0, tmp_faces.size(), [&](const uint i)
        {
            // edge_type:
            //      0 : Boundary
            //      1 : Inner edges (left)  (Same As Its Right)
            //      2 : Inner edges (right) (Same As Its Left)
            auto edge_type = list_edge_type[i];
            if (edge_type == 1) // Bending Edges (Left)
                return 1;
            else 
                return 0;
        }, 
        [&](const uint i, const uint& prefix, const uint& curr_return)
        {
            if (curr_return == 1)
            {
                const uint eid = prefix - 1;
                const Int3 curr_value = tmp_faces[i];     const Int3 curr_face = input_faces[curr_value[2]]; 
                const Int3 next_value = tmp_faces[i + 1]; const Int3 next_face = input_faces[next_value[2]]; 
                const Int2 dehedral_edge = makeInt2(curr_value[0], curr_value[1]);
                const uint curr_rest_vid = (curr_face[0] + curr_face[1] + curr_face[2]) - (dehedral_edge[0] + dehedral_edge[1]);
                const uint next_rest_vid = (next_face[0] + next_face[1] + next_face[2]) - (dehedral_edge[0] + dehedral_edge[1]);
                output_bending_edges[eid] = makeInt4(dehedral_edge[0], dehedral_edge[1], curr_rest_vid, next_rest_vid);
            }
        }, 0u);
    }
}

        
// 不用太在意绝对路径的事情，之后ui的读取模型操作会返回模型绝对路径
// static inline std::string model_path = "/Users/huohuo/Desktop/Project/HSimulation/resources/models/";
// #define SELF_RESOURCES_PATH "C:/Users/DELL/Desktop/Projects/hsimultion/resources"

bool mesh_info_to_vector(std::string mesh_name, MeshShape& mesh);

bool read_mesh_file(std::string mesh_name, MeshShape& mesh, MeshAttrib& attrib, bool use_default);
bool read_tet_file_t(std::string mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets, const bool use_default_path);
bool read_tet_file_vtk(std::string mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets, const bool use_default_path);

bool read_mesh_file(std::string mesh_name, TriangleMeshData& meshes, bool use_default_path);
bool read_mesh_file(std::string mesh_name, std::vector<TriangleMeshData>& meshes, bool use_default);
bool read_tet_file_t(std::string mesh_name, TetrahedralMeshData& meshes, const bool use_default_path);

bool saveToOBJ_saperately(const Float3* vertices, const Int3* faces, const uint* prefix_verts, const uint* prefix_faces, const uint num_clothes, const std::string& filename, const uint frame);
bool saveToOBJ_combined(const Float3* vertices, const Int3* faces, const uint* prefix_verts, const uint* prefix_faces, const uint num_clothes, const std::string& filename, const uint frame);


};

#define solver_data_path "/Users/huohuo/Desktop/Project/HSolver/solver_data/"

template<typename T>
inline void save_to_binary(SharedArray<T>& arr, std::string name){
    std::ofstream output_stream;
    output_stream.open(solver_data_path + name + ".dat", std::ofstream::binary);
    output_stream.write(reinterpret_cast<const char*>(arr.ptr()), arr.get_byte_size());
    output_stream.close();
}

template<typename T>
inline void read_from_binary(SharedArray<T>& arr,std::string name){
    std::ifstream input_stream;
    input_stream.open(solver_data_path + name + ".dat", std::ifstream::binary);

    input_stream.seekg(0, std::ios::end); // 将文件指针移到文件末尾
    std::streampos fileSize = input_stream.tellg(); // 获取文件指针的位置，即文件大小
    input_stream.seekg(0, std::ios::beg); // 将文件指针移到文件开头

    uint length = fileSize / sizeof(T);
    arr.resize(length);
    input_stream.read(reinterpret_cast<char*>(arr.ptr()), arr.get_byte_size());
    input_stream.close();
}