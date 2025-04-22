#include "mesh_reader.h"
#include "struct_to_string.h"
#include <filesystem>


namespace SimMesh{
inline void extract_surface_face_and_vert_from_tets(
    const std::vector<Float3>& input_position,
    const std::vector<Int4>& input_tets,
    std::vector<Int3>& output_faces, std::vector<uint>& output_verts)
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
    output_faces.resize(num_surface_faces);

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
            output_faces[fid] = make_ordered_face(face, orig_tet);
            list_vert_is_on_surface[curr_value[0]] = true;
            list_vert_is_on_surface[curr_value[1]] = true;
            list_vert_is_on_surface[curr_value[2]] = true;
        }
    }, 
    0u);

    uint num_surface_verts = parallel_for_and_reduce_sum<uint>(0, num_verts, [&](const uint vid) { return list_vert_is_on_surface[vid] ? 1 : 0; });
    output_verts.resize(num_surface_verts);
    parallel_for_and_scan(0, num_verts, [&](const uint vid)
    {
        return list_vert_is_on_surface[vid] ? 1 : 0;
    }, 
    [&](const uint vid, const uint& prefix, const uint& curr_return)
    {
        if (curr_return == 1)
        {
            output_verts[prefix - 1] = vid;
        }
    }, 0u);
}

inline void extract_edges_from_surface(
    const std::vector<Int3>& input_faces,
    TriangleMeshData& mesh_data)
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

    uint num_edges = 0; uint num_bending_edges = 0; uint num_boundary_edges = 0;
    for (const auto& value : list_edge_type) 
    { 
        if (value == 0) num_boundary_edges++; 
        if (value == 1) num_bending_edges++; 
        if (value == 0 || value == 1) num_edges++; 
    }
    mesh_data.edges.resize(num_edges);  
    mesh_data.bending_edges.resize(num_bending_edges);
    mesh_data.boundary_edges.resize(num_boundary_edges);

    mesh_data.edge_adj_faces.resize(num_edges);
    mesh_data.bending_edge_adj_faces.resize(num_bending_edges);

    // uint eid = 0;
    // uint eid_bending = 0;
    // uint eid_boundary = 0;
    // for (uint index = 0; index < tmp_faces.size(); index++)
    // {
    //     const Int3 curr_value = tmp_faces[index];
    //     const Int2 edge = makeInt2(curr_value[0], curr_value[1]);
    //     const uint curr_adj_fid = curr_value[2];
    //     const Int3 curr_adj_face = input_faces[curr_adj_fid]; 

    //     uint edge_type = list_edge_type[index];

    //     // edge_type:
    //     //      0 : Boundary
    //     //      1 : Inner edges (left)  (Same As Its Right)
    //     //      2 : Inner edges (right) (Same As Its Left)
        
    //     const bool is_boundary_edge = edge_type == 0;
    //     const bool is_bending_edge = edge_type == 1;
    //     const bool is_edge = is_boundary_edge || is_bending_edge;
        
    //     if (is_boundary_edge)
    //     {
    //         mesh_data.boundary_edges[eid_boundary] = edge;
    //         eid_boundary++;
    //     }
    //     if (is_bending_edge)
    //     {
    //         const Int3 next_value = tmp_faces[index + 1]; 
    //         const uint next_adj_fid = next_value[2];
    //         const Int3 next_adj_face = input_faces[next_adj_fid]; 

    //         const Int2 dehedral_edge = makeInt2(curr_value[0], curr_value[1]);
    //         const uint curr_rest_vid = (curr_adj_face[0] + curr_adj_face[1] + curr_adj_face[2]) - (dehedral_edge[0] + dehedral_edge[1]);
    //         const uint next_rest_vid = (next_adj_face[0] + next_adj_face[1] + next_adj_face[2]) - (dehedral_edge[0] + dehedral_edge[1]);
    //         mesh_data.bending_edges[eid_bending] = makeInt4(dehedral_edge[0], dehedral_edge[1], curr_rest_vid, next_rest_vid);
    //         mesh_data.bending_edge_adj_faces[eid_bending] = makeInt2(curr_adj_fid, next_adj_fid);
    //         eid_bending++;
    //     }
    //     if (is_edge)
    //     {
    //         mesh_data.edges[eid] = edge; // fast_print_single(SimString::Vec2_to_string(edge));
    //         if (is_boundary_edge)
    //         {
    //             const Int2 adj_faces = makeInt2(curr_adj_fid);
    //             mesh_data.edge_adj_faces[eid] = adj_faces;
    //         }
    //         else
    //         {
    //             const Int3 next_value = tmp_faces[index + 1]; 
    //             const uint next_adj_fid = next_value[2];
    //             const Int2 adj_faces = makeInt2(curr_adj_fid, next_adj_fid);
    //             mesh_data.edge_adj_faces[eid] = adj_faces;
    //         }
    //         eid++;
    //     }
    // }
    // fast_format("   Last eid = {} desire for {}", eid, num_edges);
    // for (auto& edge : mesh_data.edges) fast_print_single(SimString::Vec2_to_string(edge));;

    using EdgeScanType = Int3;

    const uint blockDim = 256;
    uint start_dispatch = 0 / blockDim;
    uint end_dispatch = (tmp_faces.size() + blockDim - 1) / blockDim;
    tbb::parallel_scan(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), makeInt3(0), 
        [&]( tbb::blocked_range<uint> r, EdgeScanType block_prefix, auto is_final_scan) -> EdgeScanType 
        {
            uint start_blockIdx = r.begin();
            uint end_blockIdx = r.end() - 1;
            uint startIdx = max_scalar(blockDim * start_blockIdx, 0);
            uint endIdx   = min_scalar(blockDim * (end_blockIdx + 1), tmp_faces.size());

            for (uint index = startIdx; index < endIdx; index++) 
            {
                const Int3 curr_value = tmp_faces[index];
                const Int2 edge = makeInt2(curr_value[0], curr_value[1]);
                const uint curr_adj_fid = curr_value[2];
                const Int3 curr_adj_face = input_faces[curr_adj_fid]; 

                EdgeScanType parallel_result = makeInt3(0);
                {
                    // edge_type:
                    //      0 : Boundary
                    //      1 : Inner edges (left)  (Same As Its Right)
                    //      2 : Inner edges (right) (Same As Its Left)
                    uint edge_type = list_edge_type[index];
                    if (edge_type == 0)                     parallel_result[0] = 1; // 0 -> boundary edge
                    if (edge_type == 1)                     parallel_result[1] = 1; // 1 -> bending edge
                    if (edge_type == 0 || edge_type == 1)   parallel_result[2] = 1; // 2 -> edge
                }
                block_prefix += parallel_result;
                if (is_final_scan) 
                {
                    // func_output(index, block_prefix, parallel_result);
                    {
                        if (parallel_result[0] == 1)
                        {
                            const uint boundary_eid = block_prefix[0] - 1;
                            mesh_data.boundary_edges[boundary_eid] = edge;
                        }
                        if (parallel_result[1] == 1)
                        {
                            const uint bending_eid = block_prefix[1] - 1;

                            const Int3 next_value = tmp_faces[index + 1]; 
                            const uint next_adj_fid = next_value[2];
                            const Int3 next_adj_face = input_faces[next_adj_fid]; 

                            const Int2 dehedral_edge = makeInt2(curr_value[0], curr_value[1]);
                            const uint curr_rest_vid = (curr_adj_face[0] + curr_adj_face[1] + curr_adj_face[2]) - (dehedral_edge[0] + dehedral_edge[1]);
                            const uint next_rest_vid = (next_adj_face[0] + next_adj_face[1] + next_adj_face[2]) - (dehedral_edge[0] + dehedral_edge[1]);
                            mesh_data.bending_edges[bending_eid] = makeInt4(dehedral_edge[0], dehedral_edge[1], curr_rest_vid, next_rest_vid);
                            mesh_data.bending_edge_adj_faces[bending_eid] = makeInt2(curr_adj_fid, next_adj_fid);
                        }
                        if (parallel_result[2] == 1)
                        {
                            const uint eid = block_prefix[2] - 1;
                            mesh_data.edges[eid] = edge;
                            const bool is_boundary = parallel_result[0] == 1;
                            if (is_boundary)
                            {
                                const Int2 adj_faces = makeInt2(curr_adj_fid);
                                mesh_data.edge_adj_faces[eid] = adj_faces;
                            }
                            else
                            {
                                const Int3 next_value = tmp_faces[index + 1]; 
                                const uint next_adj_fid = next_value[2];
                                const Int2 adj_faces = makeInt2(curr_adj_fid, next_adj_fid);
                                mesh_data.edge_adj_faces[eid] = adj_faces;
                            }
                        }
                    }
                }
            }
            return block_prefix;
        }, 
        [](const Int3& x, const Int3& y) -> Int3 {return x + y;},
        tbb::simple_partitioner{} );
}


bool read_mesh_file(std::string mesh_name, TriangleMeshData& mesh_data, bool use_default)
{
    std::string err, warn;


    std::string full_path;
    if(use_default)
        full_path = std::string(SELF_RESOURCES_PATH) + std::string("/models/") + mesh_name;
    else
        full_path = mesh_name;

    std::string mtl_path = std::filesystem::path(full_path).replace_extension(".mtl").string();
    

    tinyobj::ObjReader reader; tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = std::filesystem::path(full_path).parent_path().string();
    if (!reader.ParseFromFile(full_path, reader_config)) 
    {
        if (!reader.Warning().empty()) 
        {
            fast_format_err("Warning : {}", reader.Warning());
        }
        if (!reader.Error().empty()) 
        {
            fast_format_err("Warning : {}", reader.Error());
        }
        exit(1);
    }


    MeshAttrib mesh_attrib = reader.GetAttrib();
    MeshShape mesh_shape = reader.GetShapes();
    MeshMat material = reader.GetMaterials();
    
    
    const uint num_verts = static_cast<uint>(mesh_attrib.vertices.size() / 3);
    uint num_faces = 0;
    for (auto& sub_obj : mesh_shape)
    {
        num_faces += sub_obj.mesh.indices.size() / 3;
    }

    mesh_data.model_positions.resize(num_verts);
    mesh_data.faces.reserve(num_faces);
    mesh_data.normal_faces.reserve(num_faces);
    mesh_data.texcoord_faces.reserve(num_faces);
    
    mesh_data.material_ids.reserve(num_faces);
    mesh_data.material_names.reserve(material.size());

    parallel_for(0, num_verts, [&](const uint vid)
    {
        Float3 local_pos = makeFloat3(
            mesh_attrib.vertices[vid * 3 + 0], 
            mesh_attrib.vertices[vid * 3 + 1], 
            mesh_attrib.vertices[vid * 3 + 2]);
        mesh_data.model_positions[vid] = local_pos;
    });

    const bool has_uv = !mesh_attrib.texcoords.empty();
    if (has_uv)
    {
        mesh_data.has_uv = true; // fast_format(" NumUV = {}, NumVerts = {}", mesh_attrib.texcoords.size() / 2, num_verts);

        const uint num_uvs = mesh_attrib.texcoords.size() / 2;
        mesh_data.uv_positions.resize(num_uvs);
        mesh_data.uv_to_vert_map.resize(num_uvs);

        parallel_for(0, num_uvs, [&](const uint vid)
        {
            Float2 uv_pos = makeFloat2(
                mesh_attrib.texcoords[vid * 2 + 0], 
                mesh_attrib.texcoords[vid * 2 + 1]);
            mesh_data.uv_positions[vid] = uv_pos;
            mesh_data.uv_to_vert_map[vid] = vid;
        });
    }
    else 
    {
        mesh_data.has_uv = false;

        const uint num_uvs = num_verts;
        mesh_data.uv_positions.resize(num_uvs);
        mesh_data.uv_to_vert_map.resize(num_uvs);

        // Generate UV By Projection Into Diagonal Plane of AABB 
        const AABB local_aabb = parallel_for_and_reduce_sum<AABB>(0, num_verts, [&](const uint vid)
        {
            return mesh_data.model_positions[vid];
        });
        const Float3 pos_min = local_aabb.min_pos;
        const Float3 pos_max = local_aabb.max_pos;
        const Float3 pos_dim = local_aabb.range();

        struct dim_range{
            uint axis_idx;
            float axis_width;
            dim_range(uint idx, float width) : axis_idx(idx), axis_width(width) {}
        };
        dim_range tmp[3]{ dim_range(0u, pos_dim[0]), dim_range(1u, pos_dim[1]), dim_range(2u, pos_dim[2]) };
        std::sort(tmp, tmp + 3, [](const dim_range& a, const dim_range& b){
            return a.axis_width < b.axis_width;
        });
        
        Float3 tmp_e2 = Zero3;
        tmp_e2[tmp[0].axis_idx] = tmp[0].axis_width;
        tmp_e2[tmp[1].axis_idx] = tmp[1].axis_width;
        Float3 tmp_e1 = Zero3;
        tmp_e1[tmp[2].axis_idx] = tmp[2].axis_width; // 将最大的跨度作为主轴，这样不会出现投影均为0的问题

        Float3 tmp_normal = normalize_vec(cross_vec(tmp_e1, tmp_e2));
        parallel_for(0, num_verts, [&](uint vid)
        {
            Float3 pos = mesh_data.model_positions[vid];
            float distance = dot_vec(tmp_normal, pos - pos_min); // 向量由面指向点
            Float3 proj_p = pos - distance * tmp_normal;
            Float3 tmp_vec = pos - pos_min;
            float u = length_vec(project_vec(tmp_vec, tmp_e1));
            float v = length_vec(project_vec(tmp_vec, tmp_e2));

            mesh_data.uv_positions[vid] = makeFloat2(u, v);
            mesh_data.uv_to_vert_map[vid] = vid;
        });
    }

    uint face_prefix = 0;
    for (size_t submesh_idx = 0; submesh_idx < mesh_shape.size(); submesh_idx++) 
    {
        const auto& sub_mesh = mesh_shape[submesh_idx];

        auto& face_list = sub_mesh.mesh.indices;
        const uint curr_num_faces = face_list.size() / 3;
        
        for (uint fid = 0; fid < curr_num_faces; fid++)
        {
            tinyobj::index_t v0 = face_list[fid * 3 + 0];
            tinyobj::index_t v1 = face_list[fid * 3 + 1];
            tinyobj::index_t v2 = face_list[fid * 3 + 2];
            
            if (mesh_data.has_uv)
            {
                mesh_data.uv_to_vert_map[v0.texcoord_index] = v0.vertex_index;
                mesh_data.uv_to_vert_map[v1.texcoord_index] = v1.vertex_index;
                mesh_data.uv_to_vert_map[v2.texcoord_index] = v2.vertex_index;
            }
            
            int material_id = sub_mesh.mesh.material_ids[fid];
            {
                Int3 orig_face = makeInt3(
                    v0.vertex_index, 
                    v1.vertex_index, 
                    v2.vertex_index);

                if (orig_face[0] == orig_face[1] || orig_face[0] == orig_face[2] || orig_face[1] == orig_face[2])
                {   
                    fast_format_err("Illigal Face Input {} : {}", fid, SimString::Vec3_to_string(orig_face));  
                    mesh_data.invalid_material_ids.push_back(material_id);
                    mesh_data.invalid_faces.push_back(makeInt3(v0.vertex_index, v1.vertex_index, v2.vertex_index));
                    mesh_data.invalid_normal_faces.push_back(makeInt3(v0.normal_index, v1.normal_index, v2.normal_index));
                    mesh_data.invalid_texcoord_faces.push_back(makeInt3(v0.texcoord_index, v1.texcoord_index, v2.texcoord_index));
                    continue;
                }
                else 
                {
                    mesh_data.material_ids.push_back(material_id);
                    mesh_data.faces.push_back(makeInt3(v0.vertex_index, v1.vertex_index, v2.vertex_index));
                    mesh_data.normal_faces.push_back(makeInt3(v0.normal_index, v1.normal_index, v2.normal_index));
                    mesh_data.texcoord_faces.push_back(makeInt3(v0.texcoord_index, v1.texcoord_index, v2.texcoord_index));
                }  
            }
        }
        face_prefix += curr_num_faces;

        // std::cout << "Shape of submesh " << submesh_idx << " : " << mesh_shape[submesh_idx].name << std::endl;
    }
    {
        for (auto& mat : material)
        {
            mesh_data.material_names.push_back(mat.name);
            // fast_format("Materials : {} ", mat.name); // Can Not Read Materials That Have Several Entities, tinyobjloader Can Not Capture The Name
        }
    }

    extract_edges_from_surface(mesh_data.faces, mesh_data);
    
    // fast_format("   Readed Mesh Data {} : numSubMesh = {}, numVerts = {}, numFaces = {}, numEdges = {}, numBendingEdges = {}", 
    //     mesh_name, mesh_shape.size(), num_verts, num_faces, mesh_data.edges.size(), mesh_data.bending_edges.size());

    return true;
}
bool read_mesh_file(std::string mesh_name, std::vector<TriangleMeshData>& meshes, bool use_default)
{
    return true;
}
bool read_tet_file_t(std::string mesh_name, TetrahedralMeshData& meshes, const bool use_default_path)
{
    return true;
}




bool read_mesh_file(std::string mesh_name, MeshShape& mesh, MeshAttrib& attrib, bool use_default){
    
    std::string err, warn;
    // tinyobj::attrib_t attrib;
    MeshMat mat;

    std::string full_path;
    if(use_default)
        full_path = std::string(SELF_RESOURCES_PATH) + std::string("/models/") + mesh_name;
    else
        full_path = mesh_name;

    bool load = tinyobj::LoadObj(&attrib, &mesh, &mat, &warn, &err, full_path.c_str());

    if ( ! load ){
        std::cerr << "Error loading mesh from file " << full_path << " : " << err << warn << std::endl;
        return false;
    }
    else {
        return true;
    }
}
bool read_tet_file_t(std::string mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets, const bool use_default_path) 
{
    std::string err, warn;
    std::string full_path;
    if(use_default_path)
        full_path = std::string(SELF_RESOURCES_PATH) + std::string("/models/vtks/") + mesh_name;
    else
        full_path = mesh_name;

    bool load = true;
    {
        std::ifstream infile(full_path);
        if (!infile.is_open()) 
        {
            std::cerr << "Error opening file " << full_path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(infile, line)) 
        {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "Vertex") 
            {
                int index;
                float x, y, z;
                if (!(iss >> index >> x >> y >> z)) 
                {
                    std::cerr << "Error reading vertex data from file " << full_path << std::endl;
                    return false;
                }
                position.emplace_back(makeFloat3(x, y, z));
            }
            else if (prefix == "Tet") 
            {
                int index, i1, i2, i3, i4;
                if (!(iss >> index >> i1 >> i2 >> i3 >> i4)) 
                {
                    std::cerr << "Error reading tetrahedron data from file " << full_path << std::endl;
                    return false;
                }
                tets.emplace_back(makeInt4(i1, i2, i3, i4));
            }
        }

        infile.close();
    }
    return true;
}
bool read_tet_file_vtk(std::string file_name, std::vector<Float3>& positions, std::vector<Int4>& tets, const bool use_default_path) 
{
    std::string full_path = use_default_path ? 
        std::string(SELF_RESOURCES_PATH) + "/models/vtks/" + file_name : 
        file_name;

    std::ifstream infile(full_path);
    if (!infile.is_open()) 
    {
        std::cerr << "Error opening file: " << full_path << std::endl;
        return false;
    }

    std::string line;
    bool reading_points = false;
    bool reading_cells = false;
    size_t expected_points = 0, expected_cells = 0;

    while (std::getline(infile, line)) 
    {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "POINTS") 
        {
            // Read the number of points and data type (e.g., double/float)
            iss >> expected_points;
            std::string data_type;
            iss >> data_type;

            positions.reserve(expected_points);
            
            for (int i = 0; i < expected_points; ++i) 
            {
                float x, y, z;
                if (!(infile >> x >> y >> z)) 
                {
                    std::cerr << "Error reading vertex coordinates from file " << full_path << std::endl;
                    return false;
                }
                positions.emplace_back(makeFloat3(x, y, z));  // 假设 makeFloat3 用于将坐标存储为 Float3 类型
            }

        } 
        else if (keyword == "CELLS") 
        {
            // Read the number of cells and total numbers of indices
            iss >> expected_cells;
            size_t total_indices;
            iss >> total_indices;

            for (int i = 0; i < expected_cells; ++i) 
            {
                int num_points_in_cell;  // 四面体有 4 个顶点
                int p1, p2, p3, p4;
                if (!(infile >> num_points_in_cell >> p1 >> p2 >> p3 >> p4)) 
                {
                    std::cerr << "Error reading cell data from file " << full_path << std::endl;
                    return false;
                }
                // 检查四面体顶点数是否为 4
                if (num_points_in_cell != 4) 
                {
                    std::cerr << "Invalid number of points in cell " << i << std::endl;
                    return false;
                }
                tets.emplace_back(makeInt4(p1, p2, p3, p4));  // 假设 makeInt4 用于存储四面体索引
            }
        } 
        else if (keyword == "CELL_TYPES") 
        {
            // Stop reading as we no longer need the cell types for tetrahedra
            break;
        } 
    }

    infile.close();

    if (positions.empty() || tets.empty()) { fast_format("Reading Result is Empty!!! Actual Get {} Verts And {} Tetrahedrals", positions.size(), tets.size()); exit(0); }

    // fast_format("   Readed Tetrahedral Data {} : numVerts = {}, numFaces = {}, numEdges = {}, numBendingEdges = {}", 
    //     file_name, positions.size(), , num_faces, mesh_data.edges.size(), mesh_data.bending_edges.size());

    return true;
}
bool saveToOBJ_saperately(const Float3* vertices, const Int3* faces, const uint* prefix_verts, const uint* prefix_faces, const uint num_clothes, const std::string& filename, const uint frame) {

    for (uint clothIdx = 0; clothIdx < num_clothes; clothIdx++) {
        std::string full_path;

        if (num_clothes == 1) 
        {
            full_path = std::string(SELF_RESOURCES_PATH) + std::string("/output/") + 
                    filename + "_" + std::to_string(frame) + ".obj" ;
        }
        else {
            full_path = std::string(SELF_RESOURCES_PATH) + std::string("/output/") + 
                    filename + "_" + std::to_string(frame) + "_" + std::to_string(clothIdx) + ".obj" ;
        }

        std::ofstream file(full_path, std::ios::out);
        if (file.is_open()) {

            AABB tmp;
            for(uint vid = prefix_verts[clothIdx]; vid < prefix_verts[clothIdx + 1]; vid++){  tmp += vertices[vid]; }
            // std::format("[{}, {}, {}] to [{}, {}, {}]", );

            file << "# File Simulated From <Heterogeneous Cloth Simulation>" << std::endl;
            file << "# " << prefix_verts[clothIdx + 1] - prefix_verts[clothIdx] << " points" << std::endl;
            file << "# " << 3 * (prefix_faces[clothIdx + 1] - prefix_faces[clothIdx]) << " vertices" << std::endl;
            file << "# " << (prefix_faces[clothIdx + 1] - prefix_faces[clothIdx]) << " primitives" << std::endl;
            file << "# Bounds: [" << tmp.min_pos.x << ", " << tmp.min_pos.y << ", " << tmp.min_pos.z << "] to [" 
                                  << tmp.max_pos.x << ", " << tmp.max_pos.y << ", " << tmp.max_pos.z << "]" << std::endl;

            // fast_print("Vert", clothIdx, prefix_verts[clothIdx], prefix_verts[clothIdx + 1]);
            file << "g " << std::endl;

            for(uint vid = prefix_verts[clothIdx]; vid < prefix_verts[clothIdx + 1]; vid++){
                const Float3 vertex = vertices[vid];
                file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
            }

            // fast_print("Face", clothIdx, prefix_faces[clothIdx], prefix_faces[clothIdx + 1]);
            file << "g " << std::endl;

            for(uint fid = prefix_faces[clothIdx]; fid < prefix_faces[clothIdx + 1]; fid++){
                const Int3 f = faces[fid] + makeInt3(1, 1, 1) - prefix_verts[clothIdx];
                file << "f " << f.x << " " << f.y << " " << f.z << std::endl;
            }
            file.close();
            std::cout << "OBJ file saved: " << full_path << std::endl;  
        } 
        else {
            std::cerr << "Unable to open file: " << full_path << std::endl;
            return false;
        }
    }
    return true;
}
bool saveToOBJ_combined(const Float3* vertices, const Int3* faces, const uint* prefix_verts, const uint* prefix_faces, const uint num_clothes, const std::string& filename, const uint frame) {

    std::string full_path = std::string(SELF_RESOURCES_PATH) + std::string("/output/") + filename + "_" + std::to_string(frame) + ".obj";
    std::ofstream file(full_path, std::ios::out);

    if (file.is_open()) {
        file << "# File Simulated From <Heterogeneous Cloth Simulation>" << std::endl;
        
        for (uint clothIdx = 0; clothIdx < num_clothes; clothIdx++) {
            file << "o cloth_" << clothIdx << std::endl;
            file << "# " << prefix_verts[clothIdx + 1] - prefix_verts[clothIdx] << " points" << std::endl;
            file << "# " << 3 * (prefix_faces[clothIdx + 1] - prefix_faces[clothIdx]) << " vertices" << std::endl;
            file << "# " << (prefix_faces[clothIdx + 1] - prefix_faces[clothIdx]) << " primitives" << std::endl;
            for (uint vid = prefix_verts[clothIdx]; vid < prefix_verts[clothIdx + 1]; vid++) {
                const Float3 vertex = vertices[vid];
                file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
            }

            for (uint fid = prefix_faces[clothIdx]; fid < prefix_faces[clothIdx + 1]; fid++) {
                const Int3 f = faces[fid] + makeInt3(1, 1, 1);
                file << "f " << f.x << " " << f.y << " " << f.z << std::endl;
            }
        }
        file.close();
        std::cout << "OBJ file saved: " << full_path << std::endl;
        std::cout << "mesh_prefix = [";
        for (uint clothIdx = 0; clothIdx < num_clothes; clothIdx++) { std::cout << prefix_verts[clothIdx] << ", "; } 
        std::cout << prefix_verts[num_clothes] << "]" << std::endl;
            
        return true;
    } else {
        std::cerr << "Unable to open file: " << full_path << std::endl;
        return false;
    }
}

// bool read_mesh_file(std::string mesh_name, MeshInfo& mesh, bool use_default){
//     OpenMesh::IO::Options opt;
//     std::string full_path;
//     if(use_default)
//         full_path = std::string(SELF_RESOURCES_PATH) + std::string("/models/") + mesh_name;
//     else
//         full_path = mesh_name;

//     if ( ! OpenMesh::IO::read_mesh(mesh, full_path, opt)){
//         std::cerr << "Error loading mesh from file " << full_path << std::endl;
//         return false;
//     }
//     else {
//         return true;
//     }
// }



};

// //#####################################################################
// // Function Read_TriMesh_Obj
// //#####################################################################
// template <class T, int dim>
// Array<int, 4> Read_TriMesh_Obj(const std::string& filePath, 
//     Vector3f& X, Vector2i& triangles)
// {
//     std::ifstream is(filePath);
//     if (!is.is_open()) {
//         puts((filePath + " not found!").c_str());
//         return Array<int, 4>(-1, -1, -1, -1);
//     }

//     std::string line;
//     Array<T, dim> position;
//     Array<int, 3> tri;
//     Array<int, 4> counter(X.size, triangles.size, 0, 0);

//     while (std::getline(is, line)) {
//         std::stringstream ss(line);
//         if (line[0] == 'v' && line[1] == ' ') {
//             ss.ignore();
//             for (size_t i = 0; i < dim; i++)
//                 ss >> position(i);
//             X.Append(position);
//         }
//         else if (line[0] == 'f') {
//             int cnt = 0;
//             int length = line.size();
//             for (int i = 0; i < 3; ++i) {
//                 while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
//                     cnt++;
//                 int index = 0;
//                 while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
//                     index = index * 10 + line[cnt] - '0';
//                     cnt++;
//                 }
//                 tri(i) = index - 1;
//                 while (cnt < length && line[cnt] != ' ')
//                     cnt++;
//             }

//             for (int i = 0; i < 3; ++i) {
//                 tri[i] += counter[0];
//             }
//             triangles.Append(tri);

//             while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
//                 cnt++;
//             if (cnt < length) {
//                 // quad
//                 int index = 0;
//                 while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
//                     index = index * 10 + line[cnt] - '0';
//                     cnt++;
//                 }
//                 triangles.Append(Array<int, 3>(tri[0], tri[2], index - 1 + counter[0]));
//             }
//         }
//     }

//     is.close();

//     counter(2) = X.size;
//     counter(3) = triangles.size;

//     return counter;
// }

// template <class T, int dim>
// Array<int, 4> Read_TriMesh_Tex_Obj(const std::string& filePath, 
//     MESH_NODE<T, dim>& X, MESH_NODE<T, dim>& X_tex, MESH_ELEM<2>& triangles, MESH_ELEM<2>& triangles_tex,
//     std::vector<Array<int, 3>>& stitchNodes, std::vector<T>& stitchRatio)
// {
//     std::ifstream is(filePath);
//     if (!is.is_open()) {
//         puts((filePath + " not found!").c_str());
//         exit(-1);
//     }

//     std::string line;
//     Array<T, dim> position;
//     Array<int, 3> tri, tri_tex;
//     Array<int, 4> counter(X.size, triangles.size, 0, 0);
//     int texVStartInd = X_tex.size;
//     while (std::getline(is, line)) {
//         std::stringstream ss(line);
//         if (line[0] == 'v' && line[1] == ' ') {
//             ss.ignore();
//             for (size_t i = 0; i < dim; i++)
//                 ss >> position(i);
//             X.Append(position);
//         }
//         else if (line[0] == 'v' && line[1] == 't') {
//             ss.ignore(2);
//             for (size_t i = 0; i < 2; i++)
//                 ss >> position(i);
//             position[2] = 0;
//             X_tex.Append(position);
//         }
//         else if (line[0] == 'f') {
//             int cnt = 0;
//             int length = line.size();
//             bool texIndDiff = false;
//             for (int i = 0; i < 3; ++i) {
//                 while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
//                     cnt++;
//                 int index = 0;
//                 while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
//                     index = index * 10 + line[cnt] - '0';
//                     cnt++;
//                 }
//                 tri(i) = index - 1;
//                 while (cnt < length && line[cnt] != ' ' && line[cnt] != '/')
//                     cnt++;
                
//                 if(line[cnt] == '/') {
//                     cnt++;
//                     if (line[cnt] != '/') {
//                         // texture face
//                         texIndDiff = true;
//                         int index = 0;
//                         while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
//                             index = index * 10 + line[cnt] - '0';
//                             cnt++;
//                         }
//                         tri_tex(i) = index - 1;
//                     }

//                     while (cnt < length && line[cnt] != ' ')
//                         cnt++;
//                 }
//             }
//             for (int i = 0; i < 3; ++i) {
//                 tri[i] += counter[0];
//             }
//             triangles.Append(tri);

//             if (texIndDiff) {
//                 for (int i = 0; i < 3; ++i) {
//                     tri_tex[i] += texVStartInd;
//                 }
//                 triangles_tex.Append(tri_tex);
//             }
//             else {
//                 triangles_tex.Append(tri);
//             }
//         }
//         else if (line[0] == 's' && line[1] == 't' && line[2] == 'i' &&
//             line[3] == 't' && line[4] == 'c' && line[5] == 'h') 
//         {
//             std::string bypass;
//             ss >> bypass;
//             stitchNodes.resize(stitchNodes.size() + 1);
//             ss >> stitchNodes.back()[0] >> stitchNodes.back()[1] >> stitchNodes.back()[2];
//             stitchRatio.resize(stitchRatio.size() + 1);
//             ss >> stitchRatio.back();
//         }
//     }

//     is.close();

//     counter(2) = X.size;
//     counter(3) = triangles.size;

//     return counter;
// }



// class Triplet {
// public:
//     int key[3];

//     Triplet(const int* p_key)
//     {
//         key[0] = p_key[0];
//         key[1] = p_key[1];
//         key[2] = p_key[2];
//     }
//     Triplet(int key0, int key1, int key2)
//     {
//         key[0] = key0;
//         key[1] = key1;
//         key[2] = key2;
//     }

//     bool operator<(const Triplet& right) const
//     {
//         if (key[0] < right.key[0]) {
//             return true;
//         }
//         else if (key[0] == right.key[0]) {
//             if (key[1] < right.key[1]) {
//                 return true;
//             }
//             else if (key[1] == right.key[1]) {
//                 if (key[2] < right.key[2]) {
//                     return true;
//                 }
//             }
//         }
//         return false;
//     }
// };


// template <class T, bool mapTriVInd = true>
// void Find_Surface_TriMesh(
//     BASE_STORAGE<Array<T, 3>>& X,
//     BASE_STORAGE<Array<int, 4>>& Tet,
//     BASE_STORAGE<int>& TriVI2TetVI, BASE_STORAGE<Array<int, 3>>& Tri)
// {
//     // indexing triangle faces
//     std::map<Triplet, int> tri2Tet;
//     Tet.Each([&](int id, auto data) {
//         auto& [elemVInd] = data;
//         tri2Tet[Triplet(elemVInd(0), elemVInd(2), elemVInd(1))] = id;
//         tri2Tet[Triplet(elemVInd(0), elemVInd(3), elemVInd(2))] = id;
//         tri2Tet[Triplet(elemVInd(0), elemVInd(1), elemVInd(3))] = id;
//         tri2Tet[Triplet(elemVInd(1), elemVInd(2), elemVInd(3))] = id;
//     });

//     //TODO: parallelize
//     // extract surface triangles
//     // TODO: provide clear
//     if (Tri.size) Tri = std::move(BASE_STORAGE<Array<int, 3>>());
//     for (const auto& triI : tri2Tet) {
//         const int* triVInd = triI.first.key;
//         // find dual triangle with reversed indices:
//         auto finder = tri2Tet.find(Triplet(triVInd[2], triVInd[1], triVInd[0]));
//         if (finder == tri2Tet.end()) {
//             finder = tri2Tet.find(Triplet(triVInd[1], triVInd[0], triVInd[2]));
//             if (finder == tri2Tet.end()) {
//                 finder = tri2Tet.find(Triplet(triVInd[0], triVInd[2], triVInd[1]));
//                 if (finder == tri2Tet.end()) {
//                     Tri.Append(Array<int, 3>(triVInd[0], triVInd[1], triVInd[2]));
//                 }
//             }
//         }
//     }

//     // find surface nodes
//     std::vector<bool> isSurfNode(X.size, false);
//     for (int i = 0; i < Tri.size; ++i) {
//         auto [t] = Tri.Get(i).value();
//         isSurfNode[t(0)] = isSurfNode[t(1)] = isSurfNode[t(2)] = true;
//     }

//     // map surface nodes
//     std::vector<int> TetVI2TriVI(X.size, -1);
//     // TODO: provide clear
//     if (TriVI2TetVI.size) TriVI2TetVI = std::move(BASE_STORAGE<int>());
//     for (int i = 0; i < isSurfNode.size(); ++i) {
//         if (isSurfNode[i]) {
//             TetVI2TriVI[i] = TriVI2TetVI.size;
//             TriVI2TetVI.Append(i);
//         }
//     }
    
//     if constexpr (mapTriVInd) {
//         for (int i = 0; i < Tri.size; ++i) {
//             auto [t] = Tri.Get(i).value();
//             Tri.update(i, Array<int, 3>(TetVI2TriVI[t(0)], TetVI2TriVI[t(1)], TetVI2TriVI[t(2)]));
//         }
//     }
// }



// void Find_Boundary_Edge_And_Node(int Xsize, 
//     MESH_ELEM<2>& triangles,
//     std::vector<int>& boundaryNode,
//     std::vector<Array<int, 2>>& boundaryEdge)
// {
//     std::set<std::pair<int, int>> edgeSet;
//     triangles.Each([&](int id, auto data) {
//         auto &[elemVInd] = data;
//         edgeSet.insert(std::pair<int, int>(elemVInd[0], elemVInd[1]));
//         edgeSet.insert(std::pair<int, int>(elemVInd[1], elemVInd[2]));
//         edgeSet.insert(std::pair<int, int>(elemVInd[2], elemVInd[0]));
//     });

//     boundaryEdge.resize(0);
//     for (const auto& eI : edgeSet) {
//         if (edgeSet.find(std::pair<int, int>(eI.second, eI.first)) == edgeSet.end()) {
//             boundaryEdge.emplace_back(eI.first, eI.second);
//         }
//     }

//     std::vector<bool> isBoundaryNode(Xsize, false);
//     for (const auto& beI : boundaryEdge) {
//         isBoundaryNode[beI[0]] = isBoundaryNode[beI[1]] = true;
//     }
//     boundaryNode.resize(0);
//     for (int nI = 0; nI < isBoundaryNode.size(); ++nI) {
//         if (isBoundaryNode[nI]) {
//             boundaryNode.emplace_back(nI);
//         }
//     }
// }

// void Find_Surface_Primitives(
//     int Xsize, MESH_ELEM<2>& Tri,
//     std::vector<int>& boundaryNode,
//     std::vector<Array<int, 2>>& boundaryEdge,
//     std::vector<Array<int, 3>>& boundaryTri)
// {
//     boundaryTri.reserve(Tri.size);
//     std::set<Array<int, 2>> boundaryEdgeSet;
//     std::vector<bool> isBoundaryNode(Xsize, false);
//     Tri.Each([&](int id, auto data) {
//         auto &[triVInd] = data;
        
//         boundaryTri.emplace_back(triVInd);
        
//         auto finder = boundaryEdgeSet.find(Array<int, 2>(triVInd[1], triVInd[0]));
//         if (finder == boundaryEdgeSet.end()) {
//             boundaryEdgeSet.insert(Array<int, 2>(triVInd[0], triVInd[1]));
//         }
//         finder = boundaryEdgeSet.find(Array<int, 2>(triVInd[2], triVInd[1]));
//         if (finder == boundaryEdgeSet.end()) {
//             boundaryEdgeSet.insert(Array<int, 2>(triVInd[1], triVInd[2]));
//         }
//         finder = boundaryEdgeSet.find(Array<int, 2>(triVInd[0], triVInd[2]));
//         if (finder == boundaryEdgeSet.end()) {
//             boundaryEdgeSet.insert(Array<int, 2>(triVInd[2], triVInd[0]));
//         }

//         isBoundaryNode[triVInd[0]] = isBoundaryNode[triVInd[1]] = isBoundaryNode[triVInd[2]] = true;
//     });

//     boundaryEdge = std::move(std::vector<Array<int, 2>>(boundaryEdgeSet.begin(),
//         boundaryEdgeSet.end()));
    
//     for (int vI = 0; vI < isBoundaryNode.size(); ++vI) {
//         if (isBoundaryNode[vI]) {
//             boundaryNode.emplace_back(vI);
//         }
//     }
// }

// template <class T>
// void Find_Surface_Primitives_And_Compute_Area(
//     MESH_NODE<T, 3>& X, MESH_ELEM<2>& Tri,
//     std::vector<int>& boundaryNode,
//     std::vector<Array<int, 2>>& boundaryEdge,
//     std::vector<Array<int, 3>>& boundaryTri,
//     std::vector<T>& BNArea,
//     std::vector<T>& BEArea,
//     std::vector<T>& BTArea)
// {
//     boundaryTri.reserve(Tri.size);
//     BTArea.reserve(Tri.size);
//     std::map<Array<int, 2>, T> boundaryEdgeSet;
//     std::vector<T> isBoundaryNode(X.size, 0);
//     Tri.Each([&](int id, auto data) {
//         auto &[triVInd] = data;

//         const Array<T, 3>& v0 = std::get<0>(X.Get_Unchecked(triVInd[0]));
//         const Array<T, 3>& v1 = std::get<0>(X.Get_Unchecked(triVInd[1]));
//         const Array<T, 3>& v2 = std::get<0>(X.Get_Unchecked(triVInd[2]));
//         BTArea.emplace_back(0.5 * cross(v1 - v0, v2 - v0).length());
        
//         boundaryTri.emplace_back(triVInd);
        
//         auto finder = boundaryEdgeSet.find(Array<int, 2>(triVInd[1], triVInd[0]));
//         if (finder == boundaryEdgeSet.end()) {
//             boundaryEdgeSet[Array<int, 2>(triVInd[0], triVInd[1])] = BTArea.back() / 3;
//         }
//         else {
//             finder->second += BTArea.back() / 3;
//         }
//         finder = boundaryEdgeSet.find(Array<int, 2>(triVInd[2], triVInd[1]));
//         if (finder == boundaryEdgeSet.end()) {
//             boundaryEdgeSet[Array<int, 2>(triVInd[1], triVInd[2])] = BTArea.back() / 3;
//         }
//         else {
//             finder->second += BTArea.back() / 3;
//         }
//         finder = boundaryEdgeSet.find(Array<int, 2>(triVInd[0], triVInd[2]));
//         if (finder == boundaryEdgeSet.end()) {
//             boundaryEdgeSet[Array<int, 2>(triVInd[2], triVInd[0])] = BTArea.back() / 3;
//         }
//         else {
//             finder->second += BTArea.back() / 3;
//         }

//         isBoundaryNode[triVInd[0]] += BTArea.back() / 3;
//         isBoundaryNode[triVInd[1]] += BTArea.back() / 3;
//         isBoundaryNode[triVInd[2]] += BTArea.back() / 3;

//         BTArea.back() /= 2; // due to PT approx of \int_T PP
//     });

//     boundaryEdge.reserve(boundaryEdgeSet.size());
//     BEArea.reserve(boundaryEdgeSet.size());
//     for (const auto& i : boundaryEdgeSet) {
//         boundaryEdge.emplace_back(i.first);
//         BEArea.emplace_back(i.second / 2); // due to PE approx of \int_E PP and EE approx of \int_E PE
//     }
    
//     for (int vI = 0; vI < isBoundaryNode.size(); ++vI) {
//         if (isBoundaryNode[vI]) {
//             boundaryNode.emplace_back(vI);
//             BNArea.emplace_back(isBoundaryNode[vI]);
//         }
//     }
// }

