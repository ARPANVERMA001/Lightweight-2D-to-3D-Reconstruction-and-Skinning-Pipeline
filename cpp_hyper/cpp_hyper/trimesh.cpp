#include "trimesh.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>
#include <stdexcept>

static double dot(const std::array<double,3>& a, const std::array<double,3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static double mag2(const std::array<double,3>& v) {
    return dot(v,v);
}

static double mag(const std::array<double,3>& v) {
    return std::sqrt(mag2(v));
}

static std::array<double,3> cross(const std::array<double,3>& a, const std::array<double,3>& b) {
    return { a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0] };
}

template<typename T>
static std::vector<T> asarray(const std::vector<T>& v) {
    return v;
}

template<typename T>
static std::vector<T> array_copy(const std::vector<T>& v) {
    return v;
}

TriMesh::TriMesh() : lifetime_counter(0) {}

TriMesh::TriMesh(const TriMesh& other) {
    this->vs = other.vs;
    this->faces = other.faces;
    this->__face_normals = other.__face_normals;
    this->__face_areas = other.__face_areas;
    this->__vertex_normals = other.__vertex_normals;
    this->__vertex_areas = other.__vertex_areas;
    this->__edges = other.__edges;
    this->__halfedges = other.__halfedges;
    this->__vertex_halfedges = other.__vertex_halfedges;
    this->__face_halfedges = other.__face_halfedges;
    this->__edge_halfedges = other.__edge_halfedges;
    this->__directed_edge2he_index = other.__directed_edge2he_index;
    this->lifetime_counter = other.lifetime_counter;
}

TriMesh& TriMesh::operator=(const TriMesh& other) {
    if(this == &other) return *this;
    this->vs = other.vs;
    this->faces = other.faces;
    this->__face_normals = other.__face_normals;
    this->__face_areas = other.__face_areas;
    this->__vertex_normals = other.__vertex_normals;
    this->__vertex_areas = other.__vertex_areas;
    this->__edges = other.__edges;
    this->__halfedges = other.__halfedges;
    this->__vertex_halfedges = other.__vertex_halfedges;
    this->__face_halfedges = other.__face_halfedges;
    this->__edge_halfedges = other.__edge_halfedges;
    this->__directed_edge2he_index = other.__directed_edge2he_index;
    this->lifetime_counter = other.lifetime_counter;
    return *this;
}

TriMesh TriMesh::copy() const {
    return TriMesh(*this);
}

void TriMesh::update_face_normals_and_areas() {
    __face_normals.resize(faces.size());
    __face_areas.resize(faces.size());

    for (size_t f=0; f<faces.size(); f++) {
        const auto& face = faces[f];
        std::array<double,3> v0 = vs[face[0]];
        std::array<double,3> v1 = vs[face[1]];
        std::array<double,3> v2 = vs[face[2]];

        std::array<double,3> n = cross(
            std::array<double,3>{v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]},
            std::array<double,3>{v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]}
        );
        double nmag = mag(n);
        __face_normals[f] = {n[0]/nmag, n[1]/nmag, n[2]/nmag};
        __face_areas[f] = 0.5 * nmag;
    }

    assert(faces.size() == __face_normals.size());
    assert(faces.size() == __face_areas.size());
}

const std::vector<std::array<double,3>>& TriMesh::get_face_normals() {
    if(__face_normals.empty() && !faces.empty()) update_face_normals_and_areas();
    return __face_normals;
}

const std::vector<double>& TriMesh::get_face_areas() {
    if(__face_areas.empty() && !faces.empty()) update_face_normals_and_areas();
    return __face_areas;
}

void TriMesh::update_vertex_normals() {
    __vertex_normals.resize(vs.size());

    if(__face_normals.empty()) update_face_normals_and_areas();
    if(__face_areas.empty()) update_face_normals_and_areas();

    for (size_t i=0; i<__vertex_normals.size(); i++) {
        __vertex_normals[i] = {0.0,0.0,0.0};
    }

    for (size_t f=0; f<faces.size(); f++) {
        std::array<double,3> fn = {
            __face_normals[f][0]*__face_areas[f],
            __face_normals[f][1]*__face_areas[f],
            __face_normals[f][2]*__face_areas[f]
        };
        __vertex_normals[faces[f][0]][0] += fn[0];
        __vertex_normals[faces[f][0]][1] += fn[1];
        __vertex_normals[faces[f][0]][2] += fn[2];

        __vertex_normals[faces[f][1]][0] += fn[0];
        __vertex_normals[faces[f][1]][1] += fn[1];
        __vertex_normals[faces[f][1]][2] += fn[2];

        __vertex_normals[faces[f][2]][0] += fn[0];
        __vertex_normals[faces[f][2]][1] += fn[1];
        __vertex_normals[faces[f][2]][2] += fn[2];
    }

    for (size_t i=0; i<__vertex_normals.size(); i++) {
        double m = mag(__vertex_normals[i]);
        __vertex_normals[i][0] /= m;
        __vertex_normals[i][1] /= m;
        __vertex_normals[i][2] /= m;
    }

    assert(vs.size() == __vertex_normals.size());
}

const std::vector<std::array<double,3>>& TriMesh::get_vertex_normals() {
    if(__vertex_normals.empty() && !vs.empty()) update_vertex_normals();
    return __vertex_normals;
}

void TriMesh::update_vertex_areas() {
    __vertex_areas.resize(vs.size(),0.0);
    if(__face_areas.empty()) update_face_normals_and_areas();

    for (size_t i=0; i<vs.size(); i++) {
        __vertex_areas[i]=0.0;
    }

    for (size_t f=0; f<faces.size(); f++) {
        double fa = __face_areas[f];
        __vertex_areas[faces[f][0]] += fa;
        __vertex_areas[faces[f][1]] += fa;
        __vertex_areas[faces[f][2]] += fa;
    }

    for (size_t i=0; i<vs.size(); i++) {
        __vertex_areas[i] /= 3.0;
    }

    assert(vs.size() == __vertex_areas.size());
}

const std::vector<double>& TriMesh::get_vertex_areas() {
    if(__vertex_areas.empty() && !vs.empty()) update_vertex_areas();
    return __vertex_areas;
}

void TriMesh::update_edge_list() {
    std::set<std::pair<int,int>> edges;
    for (auto& face : faces) {
        {
            int i = face[0], j = face[1];
            if(i<j) edges.insert(std::make_pair(i,j));
            else edges.insert(std::make_pair(j,i));
        }
        {
            int i = face[1], j = face[2];
            if(i<j) edges.insert(std::make_pair(i,j));
            else edges.insert(std::make_pair(j,i));
        }
        {
            int i = face[2], j = face[0];
            if(i<j) edges.insert(std::make_pair(i,j));
            else edges.insert(std::make_pair(j,i));
        }
    }

    __edges.clear();
    for (auto &e : edges) {
        __edges.push_back({e.first,e.second});
    }
}

const std::vector<std::array<int,2>>& TriMesh::get_edges() {
    if(__edges.empty() && !faces.empty()) update_edge_list();
    return __edges;
}

void TriMesh::update_halfedges() {
    __halfedges.clear();
    __vertex_halfedges.clear();
    __face_halfedges.clear();
    __edge_halfedges.clear();
    __directed_edge2he_index.clear();

    if(__edges.empty() && !faces.empty()) update_edge_list();

    std::map<std::pair<int,int>, int> __directed_edge2face_index;
    for (size_t fi=0; fi<faces.size(); fi++) {
        auto face = faces[fi];
        __directed_edge2face_index[std::make_pair(face[0],face[1])] = (int)fi;
        __directed_edge2face_index[std::make_pair(face[1],face[2])] = (int)fi;
        __directed_edge2face_index[std::make_pair(face[2],face[0])] = (int)fi;
    }

    auto directed_edge2face_index = [&](const std::pair<int,int>& edge)->int {
        auto it = __directed_edge2face_index.find(edge);
        if(it != __directed_edge2face_index.end()) {
            return it->second;
        } else {
            auto rev = std::make_pair(edge.second,edge.first);
            assert(__directed_edge2face_index.find(rev) != __directed_edge2face_index.end());
            return -1;
        }
    };

    __vertex_halfedges.resize(vs.size(), -1);
    __face_halfedges.resize(faces.size(), -1);
    __edge_halfedges.resize(__edges.size(), -1);

    for (size_t ei=0; ei<__edges.size(); ei++) {
        auto edge = __edges[ei];
        HalfEdge he0;
        he0.face = directed_edge2face_index(std::make_pair(edge[0],edge[1]));
        he0.to_vertex = edge[1];
        he0.edge = (int)ei;

        HalfEdge he1;
        he1.face = directed_edge2face_index(std::make_pair(edge[1],edge[0]));
        he1.to_vertex = edge[0];
        he1.edge = (int)ei;

        int he0index = (int)__halfedges.size();
        __halfedges.push_back(he0);
        int he1index = (int)__halfedges.size();
        __halfedges.push_back(he1);

        __halfedges[he0index].opposite_he = he1index;
        __halfedges[he1index].opposite_he = he0index;

        __directed_edge2he_index[std::make_pair(edge[0],edge[1])] = he0index;
        __directed_edge2he_index[std::make_pair(edge[1],edge[0])] = he1index;

        if(__vertex_halfedges[__halfedges[he0index].to_vertex] == -1 || he1.face == -1) {
            __vertex_halfedges[__halfedges[he0index].to_vertex] = __halfedges[he0index].opposite_he;
        }
        if(__vertex_halfedges[__halfedges[he1index].to_vertex] == -1 || he0.face == -1) {
            __vertex_halfedges[__halfedges[he1index].to_vertex] = __halfedges[he1index].opposite_he;
        }

        if(he0.face != -1 && __face_halfedges[he0.face] == -1) {
            __face_halfedges[he0.face] = he0index;
        }
        if(he1.face != -1 && __face_halfedges[he1.face] == -1) {
            __face_halfedges[he1.face] = he1index;
        }

        __edge_halfedges[ei] = he0index;
    }

    std::vector<int> boundary_heis;
    for (size_t hei=0; hei<__halfedges.size(); hei++) {
        if(__halfedges[hei].face == -1) {
            boundary_heis.push_back((int)hei);
            continue;
        }

        int f = __halfedges[hei].face;
        auto face = faces[f];

        int tv = __halfedges[hei].to_vertex;
        int idx = -1;
        for (int k=0; k<3; k++) {
            if(face[k]==tv) {idx=k;break;}
        }
        int j = face[(idx+1)%3];

        auto it = __directed_edge2he_index.find(std::make_pair(tv,j));
        assert(it != __directed_edge2he_index.end());
        __halfedges[hei].next_he = it->second;
    }

    std::map<int,std::set<int>> vertex2outgoing_boundary_hei;
    for (auto hei : boundary_heis) {
        int originating_vertex = __halfedges[__halfedges[hei].opposite_he].to_vertex;
        vertex2outgoing_boundary_hei[originating_vertex].insert(hei);
    }

    for (auto hei : boundary_heis) {
        int tov = __halfedges[hei].to_vertex;
        auto &outset = vertex2outgoing_boundary_hei[tov];
        if(outset.empty()) {
            // pass
        } else {
            int outgoing_hei = *outset.begin();
            outset.erase(outset.begin());
            __halfedges[hei].next_he = outgoing_hei;
        }
    }
}

int TriMesh::directed_edge2he_index(const std::pair<int,int>& edge) {
    if(__directed_edge2he_index.empty() && !faces.empty()) update_halfedges();
    auto it = __directed_edge2he_index.find(edge);
    if(it == __directed_edge2he_index.end()) {
        throw std::runtime_error("Edge not found");
    }
    return it->second;
}

const std::vector<TriMesh::HalfEdge>& TriMesh::halfedges() {
    if(__halfedges.empty() && !faces.empty()) update_halfedges();
    return __halfedges;
}

std::vector<int> TriMesh::vertex_vertex_neighbors(int vertex_index) {
    if(__halfedges.empty() && !faces.empty()) update_halfedges();
    std::vector<int> result;
    int start_he = __vertex_halfedges[vertex_index];
    const HalfEdge* start = &__halfedges[start_he];
    const HalfEdge* he = start;
    do {
        result.push_back(he->to_vertex);

        const HalfEdge& opp = __halfedges[he->opposite_he];
        he = &__halfedges[opp.next_he];
    } while(he != start);
    return result;
}

int TriMesh::vertex_valence(int vertex_index) {
    auto neigh = vertex_vertex_neighbors(vertex_index);
    return (int)neigh.size();
}

std::vector<int> TriMesh::vertex_face_neighbors(int vertex_index) {
    if(__halfedges.empty() && !faces.empty()) update_halfedges();
    std::vector<int> result;
    int start_he = __vertex_halfedges[vertex_index];
    const HalfEdge* start = &__halfedges[start_he];
    const HalfEdge* he = start;
    do {
        if(he->face != -1) result.push_back(he->face);

        const HalfEdge& opp = __halfedges[he->opposite_he];
        he = &__halfedges[opp.next_he];
    } while(he != start);
    return result;
}

bool TriMesh::vertex_is_boundary(int vertex_index) {
    if(__halfedges.empty() && !faces.empty()) update_halfedges();
    int he_i = __vertex_halfedges[vertex_index];
    return (__halfedges[he_i].face == -1);
}

std::vector<int> TriMesh::boundary_vertices() {
    if(__halfedges.empty() && !faces.empty()) update_halfedges();
    std::set<int> result_set;
    for (size_t hei=0; hei<__halfedges.size(); hei++) {
        if(__halfedges[hei].face == -1) {
            int v1 = __halfedges[hei].to_vertex;
            int v2 = __halfedges[__halfedges[hei].opposite_he].to_vertex;
            result_set.insert(v1);
            result_set.insert(v2);
        }
    }
    std::vector<int> result(result_set.begin(), result_set.end());
    return result;
}

std::vector<std::pair<int,int>> TriMesh::boundary_edges() {
    if(__halfedges.empty() && !faces.empty()) update_halfedges();
    std::vector<std::pair<int,int>> result;
    for (size_t hei=0; hei<__halfedges.size(); hei++) {
        if(__halfedges[hei].face == -1) {
            const HalfEdge& he = __halfedges[hei];
            int ov = __halfedges[he.opposite_he].to_vertex;
            result.push_back(std::make_pair(ov, he.to_vertex));
        }
    }
    return result;
}

void TriMesh::positions_changed() {
    __face_normals.clear();
    __face_areas.clear();
    __vertex_normals.clear();
    __vertex_areas.clear();

    lifetime_counter += 1;
}

void TriMesh::topology_changed() {
    __edges.clear();
    __halfedges.clear();
    __vertex_halfedges.clear();
    __face_halfedges.clear();
    __edge_halfedges.clear();
    __directed_edge2he_index.clear();

    positions_changed();
}

std::vector<int> TriMesh::get_dangling_vertices() {
    if(faces.empty()) {
        std::vector<int> result(vs.size());
        for (size_t i=0; i<vs.size(); i++) result[i]=(int)i;
        return result;
    }
    std::vector<bool> vertex_has_face(vs.size(), false);
    for (auto& f : faces) {
        vertex_has_face[f[0]] = true;
        vertex_has_face[f[1]] = true;
        vertex_has_face[f[2]] = true;
    }
    std::vector<int> dang;
    for (size_t i=0; i<vs.size(); i++) {
        if(!vertex_has_face[i]) dang.push_back((int)i);
    }
    return dang;
}

std::vector<int> TriMesh::remove_vertex_indices(const std::vector<int>& vertex_indices_to_remove) {
    if(vertex_indices_to_remove.empty()) {
        std::vector<int> old2new(vs.size());
        for (size_t i=0; i<vs.size(); i++) old2new[i]=(int)i;
        return old2new;
    }

    std::vector<bool> removed(vs.size(),false);
    for (auto idx : vertex_indices_to_remove) {
        removed[idx]=true;
    }

    std::vector<int> keep_vertices;
    keep_vertices.reserve(vs.size());
    for (size_t i=0; i<vs.size(); i++) {
        if(!removed[i]) keep_vertices.push_back((int)i);
    }

    std::vector<int> old2new(vs.size(),-1);
    for (size_t i=0; i<keep_vertices.size(); i++) {
        old2new[ keep_vertices[i] ] = (int)i;
    }

    {
        std::vector<std::array<double,3>> new_vs;
        new_vs.resize(keep_vertices.size());
        for (size_t i=0; i<keep_vertices.size(); i++) {
            new_vs[i] = vs[ keep_vertices[i] ];
        }
        vs = std::move(new_vs);
    }

    {
        std::vector<std::array<int,3>> new_faces;
        new_faces.reserve(faces.size());
        for (auto &f : faces) {
            int i0=old2new[f[0]];
            int i1=old2new[f[1]];
            int i2=old2new[f[2]];
            if(i0!=-1 && i1!=-1 && i2!=-1) {
                new_faces.push_back({i0,i1,i2});
            }
        }
        faces = std::move(new_faces);
    }

    topology_changed();

    auto dangling = get_dangling_vertices();
    if(!dangling.empty()) {
        auto old2new_recurse = remove_vertex_indices(dangling);
        for (size_t i=0; i<old2new.size(); i++) {
            if(old2new[i]!=-1) {
                old2new[i] = old2new_recurse[ old2new[i] ];
            }
        }
    }

    return old2new;
}

std::vector<int> TriMesh::remove_face_indices(const std::vector<int>& face_indices_to_remove) {
    if(face_indices_to_remove.empty()) {
        std::vector<int> old2new(faces.size());
        for (size_t i=0; i<faces.size(); i++) old2new[i]=(int)i;
        return old2new;
    }

    std::vector<bool> face_removed(faces.size(),false);
    for (auto idx : face_indices_to_remove) {
        face_removed[idx]=true;
    }

    int count_keep=0;
    for (bool b : face_removed) if(!b) count_keep++;
    std::vector<int> keep_faces; keep_faces.reserve(count_keep);
    for (size_t i=0; i<faces.size(); i++) {
        if(!face_removed[i]) keep_faces.push_back((int)i);
    }

    std::vector<int> old2new(faces.size(),-1);
    for (size_t i=0; i<keep_faces.size(); i++) {
        old2new[keep_faces[i]] = (int)i;
    }

    {
        std::vector<std::array<int,3>> new_faces;
        new_faces.resize(keep_faces.size());
        for (size_t i=0; i<keep_faces.size(); i++) {
            new_faces[i] = faces[ keep_faces[i] ];
        }
        faces = std::move(new_faces);
    }

    topology_changed();

    auto dangling = get_dangling_vertices();
    if(!dangling.empty()) {
        remove_vertex_indices(dangling);
    }

    return old2new;
}

void TriMesh::append(const TriMesh& mesh) {
    int vertex_offset = (int)vs.size();

    {
        size_t old_size = vs.size();
        vs.resize(old_size + mesh.vs.size());
        for (size_t i=0; i<mesh.vs.size(); i++) {
            vs[old_size+i] = mesh.vs[i];
        }
    }

    {
        size_t old_size = faces.size();
        faces.resize(old_size + mesh.faces.size());
        for (size_t i=0; i<mesh.faces.size(); i++) {
            faces[old_size+i] = {
                mesh.faces[i][0]+vertex_offset,
                mesh.faces[i][1]+vertex_offset,
                mesh.faces[i][2]+vertex_offset
            };
        }
    }

    std::vector<std::array<double,3>> old_fn = __face_normals;
    std::vector<double> old_fa = __face_areas;
    std::vector<std::array<double,3>> old_vn = __vertex_normals;
    std::vector<double> old_va = __vertex_areas;

    topology_changed();

    if(!old_fn.empty() && !mesh.__face_normals.empty()) {
        __face_normals.clear();
        __face_normals.reserve(old_fn.size() + mesh.__face_normals.size());
        for (auto &n : old_fn) __face_normals.push_back(n);
        for (auto &n : mesh.__face_normals) __face_normals.push_back(n);
    }

    if(!old_fa.empty() && !mesh.__face_areas.empty()) {
        __face_areas.clear();
        __face_areas.reserve(old_fa.size() + mesh.__face_areas.size());
        for (auto &a : old_fa) __face_areas.push_back(a);
        for (auto &a : mesh.__face_areas) __face_areas.push_back(a);
    }

    if(!old_vn.empty() && !mesh.__vertex_normals.empty()) {
        __vertex_normals.clear();
        __vertex_normals.reserve(old_vn.size() + mesh.__vertex_normals.size());
        for (auto &n : old_vn) __vertex_normals.push_back(n);
        for (auto &n : mesh.__vertex_normals) __vertex_normals.push_back(n);
    }

    if(!old_va.empty() && !mesh.__vertex_areas.empty()) {
        __vertex_areas.clear();
        __vertex_areas.reserve(old_va.size() + mesh.__vertex_areas.size());
        for (auto &a : old_va) __vertex_areas.push_back(a);
        for (auto &a : mesh.__vertex_areas) __vertex_areas.push_back(a);
    }
}

TriMesh TriMesh::FromTriMeshes(const std::vector<TriMesh>& meshes) {
    TriMesh result;
    for (auto &m : meshes) {
        result.append(m);
    }
    result.lifetime_counter=0;
    return result;
}

TriMesh TriMesh::FromOBJ_Lines(std::istream& in) {
    TriMesh result;
    std::string line;
    while(std::getline(in,line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        if(token=="v") {
            double x,y,z;
            ss >> x >> y >> z;
            result.vs.push_back({x,y,z});
        } else if(token=="f") {
            std::vector<int> face_indices;
            std::string vstr;
            while(ss >> vstr) {
                int slash_pos = (int)vstr.find('/');
                std::string v_idx_str = (slash_pos == (int)std::string::npos) ? vstr : vstr.substr(0,slash_pos);
                int idx = std::stoi(v_idx_str);
                if(idx<0) {
                    idx = (int)result.vs.size() + idx;
                } else {
                    idx = idx-1;
                }
                face_indices.push_back(idx);
            }
            assert(face_indices.size()==3);
            result.faces.push_back({face_indices[0], face_indices[1], face_indices[2]});
        }
    }

    for (auto &f : result.faces) {
        for (int i=0; i<3; i++) {
            assert(f[i]>=0 && f[i]<(int)result.vs.size());
        }
    }

    return result;
}

TriMesh TriMesh::FromOBJ_FileName(const std::string& fname) {
    std::ifstream f(fname.c_str(), std::ios::binary);
    if(!f.is_open()) {
        throw std::runtime_error("Cannot open file");
    }
    return FromOBJ_Lines(f);
}

void TriMesh::write_OBJ(const std::string& fname, const std::string& header_comment) {
    std::ofstream out(fname.c_str());
    if(!out.is_open()) {
        throw std::runtime_error("Cannot open file to write");
    }

    std::string hc = header_comment;
    if(hc.empty()) {
        hc = "Written by TriMesh C++ version";
    }

    {
        std::stringstream sss(hc);
        std::string line;
        while(std::getline(sss,line)) {
            out << "## " << line << "\n";
        }
        out << "\n";
    }

    for (auto &v : vs) {
        out << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    out << "\n";

    for (auto &f : faces) {
        out << "f " << f[0]+1 << " " << f[1]+1 << " " << f[2]+1 << "\n";
    }

    out.close();
    std::cout << "OBJ written to: " << fname << "\n";
}

void TriMesh::write_OFF(const std::string& fname) {
    std::ofstream out(fname.c_str());
    if(!out.is_open()) {
        throw std::runtime_error("Cannot open file to write");
    }

    out << "OFF\n";
    out << vs.size() << " " << faces.size() << " 0\n";
    for (auto &v : vs) {
        out << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    for (auto &f : faces) {
        out << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    out.close();
    std::cout << "OFF written to: " << fname << "\n";
}