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

// 不用太在意绝对路径的事情，之后ui的读取模型操作会返回模型绝对路径
// static inline std::string model_path = "/Users/huohuo/Desktop/Project/HSimulation/resources/models/";
// #define SELF_RESOURCES_PATH "C:/Users/DELL/Desktop/Projects/hsimultion/resources"

bool mesh_info_to_vector(std::string mesh_name, MeshShape& mesh);

bool read_mesh_file(std::string mesh_name, MeshShape& mesh, MeshAttrib& attrib, bool use_default);
bool read_tet_file_t(std::string mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets, const bool use_default_path);
bool read_tet_file_vtk(std::string mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets, const bool use_default_path);

bool read_mesh_file(std::string mesh_name, TriangleMeshData& meshes, bool use_default);
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