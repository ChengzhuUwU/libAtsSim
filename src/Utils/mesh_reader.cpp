#include "mesh_reader.h"
#include "struct_to_string.h"
#include <filesystem>


namespace SimMesh{




bool read_mesh_file(std::string mesh_name, TriangleMeshData& mesh_data, bool use_default_path)
{
    std::string err, warn;


    std::string full_path;
    if (use_default_path)
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

    extract_edges_from_surface<true>(mesh_data.faces, mesh_data.edges, mesh_data.bending_edges);
    
    // fast_format("   Readed Mesh Data {} : numSubMesh = {}, numVerts = {}, numFaces = {}, numEdges = {}, numBendingEdges = {}", 
    //     mesh_name, mesh_shape.size(), num_verts, num_faces, mesh_data.edges.size(), mesh_data.bending_edges.size());

    return true;
}
bool read_mesh_file(std::string mesh_name, std::vector<TriangleMeshData>& meshes, bool use_default)
{
    // Containing submesh...???
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

