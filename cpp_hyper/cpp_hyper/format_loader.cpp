#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <cassert>
#include <Eigen/Dense>

// Replicate the logic of removing 'n' prefix from lines as in Python code.
// In Python code: if (nline[0]=='n') { nline=nline[11:-1]; }
// We'll write a helper function to clean a line if needed:
std::string clean_nan_line(const std::string &input) {
    if (!input.empty() && input[0]=='n') {
        // In python: nline=nline[11:-1]
        // means drop first 11 chars and last 1 char
        // if input = "nan(...)" it might remove 'nan(' and ')'?
        // We'll trust the python logic exactly:
        if (input.size() > 12) {
            return input.substr(11, input.size()-11-1); 
        } else {
            // If line is too short, just return empty?
            return "";
        }
    }
    return input;
}

// load_DMAT(path):
// Reads first line for dims, then reads matrix data line by line.
// Returns Eigen::MatrixXd
Eigen::MatrixXd load_DMAT(const std::string &path) {
    std::ifstream f(path);
    if(!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::string line;
    std::vector<int> dims;
    std::vector<double> Mdata;
    int line_count=0;
    while(std::getline(f, line)) {
        if(line_count==0) {
            // first line: dims
            std::stringstream ss(line);
            int d;
            while(ss>>d) dims.push_back(d);
            if(dims.size()<2) {
                throw std::runtime_error("DMAT must have at least two dims");
            }
            Mdata.reserve((size_t)dims[0]*dims[1]);
        } else {
            std::string nline=line;
            if(!nline.empty() && nline.back()=='\n') {
                nline.pop_back();
            }
            nline = clean_nan_line(nline);
            double val=std::stod(nline);
            Mdata.push_back(val);
        }
        line_count++;
    }

    if((int)Mdata.size() != dims[0]*dims[1]) {
        throw std::runtime_error("DMAT data size does not match dims");
    }
    Eigen::MatrixXd M(dims[1], dims[0]); 
    // Note: The python code: M.reshape(dims) with dims presumably (cols, rows)
    // The code writes: f.write(cols rows) and expects M shape as (rows,cols) in memory?
    // The DMAT final shape: M is dims (???), it writes "cols rows" in write_DMAT.
    // The Python code: M = M.reshape(dims)
    // dims from the code: line 0: "cols rows"
    // In python: M = M.reshape(dims) means M.shape=(cols, rows)
    // But later write_DMAT writes (cols rows) and loops over i in range(cols), M[:,i].
    // M in python likely stored as (cols,rows) shape.
    // We want to be consistent:
    // Let's store M in Eigen as M(rows,cols) because that's natural for Eigen.
    // dims from python line: f.write(repr(cols)+" "+repr(rows))
    // So dims[0]=cols, dims[1]=rows from python code.
    // After reading, M = M.reshape( dims ) means M shaped (cols,rows)
    // We'll store as M(rows,cols) in Eigen, so we must do M.transposeInPlace() if needed.

    // Python: The final M shape is (dims), i.e. (cols, rows).
    // We want rows,cols to match normal indexing: If dims=[cols,rows],
    // Then M after reshape is M with shape=(cols,rows). We want M in Eigen as rows x cols.
    // We'll do: Mtemp is (cols x rows) from Mdata in row-major
    // We must reorder. It's simpler to just fill M as shape (dims[0], dims[1]) and then transpose.
    // Let's first create Mtemp to match python shape exactly (cols,rows):
    Eigen::MatrixXd Mtemp(dims[0], dims[1]);
    {
        int idx=0;
        for(int r=0; r<dims[0]; r++) {
            for(int c=0; c<dims[1]; c++) {
                Mtemp(r,c)=Mdata[idx++];
            }
        }
    }
    // Now Mtemp is (cols x rows), Python indexing: Mtemp.shape=(cols,rows)
    // In Python M is (cols,rows). For consistency with normal indexing (rows,cols),
    // we transpose it:
    Eigen::MatrixXd Mfinal = Mtemp.transpose();
    return Mfinal;
}

void write_DMAT(const std::string &path, const Eigen::MatrixXd &M) {
    // M is a matrix (rows x cols)
    // Python writes: f.write(cols+" "+rows+"\n")
    // and then loops i in range(cols), for e in M[:,i], write e
    // This means DMAT format is: first line: cols rows
    // data stored column-wise
    int rows = (int)M.rows();
    int cols = (int)M.cols();

    std::ofstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open file: " + path);

    f << cols << " " << rows << "\n";
    for(int i=0; i<cols; i++) {
        for (int r=0; r<rows; r++) {
            f << M(r,i) << "\n";
        }
    }
}

Eigen::MatrixXd load_Tmat(const std::string &path) {
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open file:"+path);

    std::vector<double> vals;
    std::string line;
    while(std::getline(f,line)) {
        std::stringstream ss(line);
        double v;
        while(ss>>v) {
            vals.push_back(v);
        }
    }
    // reshape into (-1,12)
    // number of rows = vals.size()/12
    if(vals.size()%12!=0) {
        throw std::runtime_error("Tmat data not divisible by 12");
    }
    int rows = (int)vals.size()/12;
    Eigen::MatrixXd M(rows,12);
    int idx=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<12;j++){
            M(i,j)=vals[idx++];
        }
    }
    return M;
}

std::vector<std::vector<std::array<double,3>>> load_poses(const std::string &path) {
    // Load first line: dims
    // Then read all other lines as floats
    // M = appended rows
    // Then M = M.reshape((dims[0], -1, 3))
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open:"+path);

    std::string line;
    std::vector<int> dims;
    std::vector<double> allvals;
    int line_count=0;
    while(std::getline(f,line)){
        if(line_count==0) {
            std::stringstream ss(line);
            int d;
            while(ss>>d) dims.push_back(d);
        } else {
            std::stringstream ss(line);
            double v;
            while(ss>>v) allvals.push_back(v);
        }
        line_count++;
    }

    if(dims.size()<1) throw std::runtime_error("load_poses: dims not found properly");
    int D0 = dims[0];
    int total = (int)allvals.size();
    // final shape: (D0, -1, 3)
    if(total%3!=0) throw std::runtime_error("total values not divisible by 3");
    int D1 = total/(D0*3);
    if(D1<=0) throw std::runtime_error("Invalid dimension in load_poses");

    // We'll create a structure: M[D0][D1][3]
    std::vector<std::vector<std::array<double,3>>> M(D0, std::vector<std::array<double,3>>(D1));
    int idx=0;
    for(int i=0;i<D0;i++){
        for(int j=0;j<D1;j++){
            std::array<double,3> arr;
            for(int k=0;k<3;k++){
                arr[k]=allvals[idx++];
            }
            M[i][j]=arr;
        }
    }
    return M;
}

std::pair<std::vector<std::vector<std::array<double,12>>>, Eigen::MatrixXd> load_result(const std::string &path){
    // M is Bone-by-Frame-by-12
    // W will store vertex weights
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open:"+path);

    std::string line;
    std::string section;
    int B=0;
    int nframes=0;
    int count=0;
    int rows=0;
    Eigen::MatrixXd W; // will define size after we know rows,B
    std::vector<std::vector<std::array<double,12>>> M; 
    // We must store per bone a vector of frames each with an array of 12 doubles.

    // We'll accumulate a temporary std::vector<std::array<double,12>> for each bone
    // When we encounter *BONEANIMATION, we start reading that bone's frames.
    // When done reading frames, we know we must next bone start.

    // Actually code appends M line by line in global M.
    // After finishing file, M = asarray(M).reshape(B,-1,12)
    // We'll first store all frames from all bones in a global vector
    std::vector<std::array<double,12>> globalBoneFrames;

    int current_bone_frames=0;
    bool reading_bone=false;
    bool reading_weight=false;
    int current_bone = -1;

    // W is read after section="weight" lines
    // first we know rows and have W sized (rows,B)
    // then fill W

    while(std::getline(f,line)) {
        if(line.empty()) continue;
        {
            // check if line is "*BONEANIMATION"
            // or "*VERTEXWEIGHTS"
            std::stringstream ss(line);
            std::vector<std::string> words;
            {
                std::string w;
                while(std::getline(ss,w,' ')) {
                    if(!w.empty()) words.push_back(w);
                }
            }

            if(words.size()==3 && words[0]=="*BONEANIMATION"){
                section="bone";
                // parse NFRAMES from words[2]. words[2] like "NFRAMES=xx"
                std::string nfr=words[2];
                int eqpos=(int)nfr.find("=");
                int nf=std::stoi(nfr.substr(eqpos+1));
                nframes=nf;
                B+=1;
                current_bone++;
                reading_bone=true;
                reading_weight=false;
                count=0;
                continue;
            }
            else if(words.size()>0 && words[0]=="*VERTEXWEIGHTS"){
                section="weight";
                reading_bone=false;
                reading_weight=true;
                // parse rows from words[1]
                // words[1] like "ROWS=xx"
                std::string r=words[1];
                int eqpos=(int)r.find("=");
                rows = std::stoi(r.substr(eqpos+1));
                // now we know rows and B (number of bones)
                // W shape (rows,B)
                W = Eigen::MatrixXd::Zero(rows,B);
                count=0;
                continue;
            }
        }

        if(section=="bone" && reading_bone && count<nframes){
            // parse a line: "FrameNumber val1 val2 ... val12 something"
            // code: assert(len(words)==17), words[1:13] are the vals
            // Let's re-split line by space:
            std::stringstream s2(line);
            std::vector<std::string> parts;
            {
                std::string w;
                while(s2>>w) parts.push_back(w);
            }
            if((int)parts.size()!=17) {
                throw std::runtime_error("Bone line does not have 17 words");
            }
            std::array<double,12> arr;
            for(int i=0;i<12;i++){
                std::string val=parts[i+1];
                // if val starts with 'n', handle as python does
                if(!val.empty() && val[0]=='n'){
                    // val=val[11:-1] in python
                    if(val.size()>12)
                        val=val.substr(11, val.size()-11-1);
                }
                arr[i]=std::stod(val);
            }
            globalBoneFrames.push_back(arr);
            count++;
        } else if(section=="weight" && reading_weight && count<rows){
            // line like: "vertex_id ... pairs of cidx val ..."
            std::stringstream s2(line);
            std::vector<std::string> parts;
            {
                std::string w;
                while(s2>>w) parts.push_back(w);
            }

            // assert(len(words)%2==0)
            // words format: ridx something pairs...
            // Actually code: assert(len(words)%2==0)
            if(parts.size()%2!=0) {
                throw std::runtime_error("Weight line does not have even number of elements");
            }
            int ridx=std::stoi(parts[0]);
            // The line structure for weights: ridx ... cidx val ...
            // According to python code:
            // for i in range(int(len(words[2:])/2)):
            // cidx=words[i*2+2], val=words[i*2+3]
            // In C++:
            int pairCount=(int)(parts.size()-2)/2;
            for(int i=0;i<pairCount;i++){
                int cidx=std::stoi(parts[i*2+2]);
                double val=std::stod(parts[i*2+3]);
                W(ridx,cidx)=val;
            }
            count++;
        }
    }

    // Now we have globalBoneFrames with B*nframes frames each with 12 vals
    // reshape: M = asarray(M).reshape(B,-1,12)
    // total frames = globalBoneFrames.size()
    // nframes known from the last bone read. 
    // But careful: each bone section sets nframes. We assume all bones have same nframes?
    // The code does: M = M.reshape(B,-1,12) after reading all?
    // B we counted incrementally, and nframes set from the last bone or all bones have same nframes?
    // Let's assume each bone had a separate nframes known.
    // Actually the code sets nframes each bone. If different bones had different nframes?
    // That would fail. We'll trust all bones have same nframes from code style.

    // total length = globalBoneFrames.size()
    // each bone has nframes frames
    // so total = B*nframes
    int total = (int)globalBoneFrames.size();
    if(total%(B)==0) {
        int frames_per_bone = total/B; 
        if(frames_per_bone*B != total) {
            throw std::runtime_error("M data not divisible by B");
        }
        // frames_per_bone should match nframes
        // Create M structure: M[B][frames_per_bone][12]
        std::vector<std::vector<std::array<double,12>>> Mout(B, std::vector<std::array<double,12>>((size_t)frames_per_bone));
        int idx=0;
        for(int b=0;b<B;b++){
            for(int fr=0;fr<frames_per_bone;fr++){
                Mout[b][fr]=globalBoneFrames[idx++];
            }
        }

        // W returned as W.T in python, original line: return M, W.T
        // If W is currently (rows,B), W.T is (B, rows)
        // Let's transpose W:
        Eigen::MatrixXd Wt = W.transpose();

        return std::make_pair(Mout, Wt);
    } else {
        throw std::runtime_error("Mismatch in bone frames count");
    }
}

void write_result(const std::string &path_res, const std::string &path_w,
                         const Eigen::MatrixXd &res, const Eigen::MatrixXd &weights,
                         int iter_num, double time, bool col_major=false) {
    // res: originally shaped as (B, nframes, 12)
    // We have res as a 2D matrix probably (B*nframes, 12) or something.
    // In python: res = res.reshape(B,-1,12)
    // Let's assume res is given already as (B*nframes,12) or (???).
    // The code calls res = res.reshape(B,-1,12).
    // We'll assume input res is shape: B*nframes x 12 (like python had after final).
    // dimension needed from python code: B and nframes
    // B * nframes = res.rows(), 12 = res.cols()

    int total_rows=(int)res.rows();
    int B; 
    int nframes;
    // We don't have B explicitly, we must guess or user ensures correct shape.
    // The python code: res is B, nframes,12. B and nframes known from context?
    // Let's just say the user must supply correct B and frames from res shape or we can't know B.
    // Without that info, we can't guess B.
    // The python code does: B = len(res), res = res.reshape(B,-1,12)
    // Here we must know B from context.
    // We'll guess B from `weights` dimension: weights shape (B, ???)
    // weights rows = B?
    int B_ = (int)weights.rows();
    if(B_>0 && total_rows% B_ ==0) {
        B=B_;
        nframes = total_rows/B;
    } else {
        throw std::runtime_error("Cannot deduce B and nframes from res and weights shape");
    }

    // If col_major: we must reinterpret and transpose
    Eigen::MatrixXd res_final = res;
    if(col_major) {
        // col_major means we must treat each 12 block as a 4x3 and transpose?
        // python code:
        // if col_major:
        //   res = res.reshape(B,-1,4,3)
        //   res = swapaxes(res,2,3)
        //   res = reshape(B,-1,12)
        // We'll replicate:
        // First reshape to (B,nframes,4,3):
        // We'll do it step by step:
        Eigen::MatrixXd tmp(B*nframes,12);
        int idx=0;
        for(int b=0;b<B;b++){
            for(int fr=0;fr<nframes;fr++){
                // extract 12 vals
                // Eigen::Matrix<double,4,3> block;
                Eigen::MatrixXd block(4,3);
                for(int val_i=0;val_i<12;val_i++){
                    block(val_i/3,val_i%3)=res(b*nframes+fr,val_i);
                }
                // swapaxes(2,3) means transpose block:
                block.transposeInPlace(); // now 3x4
                // ... fill block ...
                Eigen::MatrixXd blockT = block.transpose(); // now blockT is 3x4
                block = blockT; // block is now 3x4
                // flatten back to 12:
                // now block is 3x4, we must store as row-major 12:
                // after transpose we got (3x4). python final shape is (B,-1,12) means still row-major?
                // In python: swapaxes just changes dimension order. The final flatten is the same order='C'.
                // So final is (4,3)->(3,4). Just read row major from 3x4:
                // Actually original block was (4x3), after transpose is (3x4). We must store back as a 1D 12 vector:
                int jdx=0;
                for(int rr=0;rr<3;rr++){
                    for(int cc=0;cc<4;cc++){
                        tmp(b*nframes+fr,jdx++)=block(rr,cc);
                    }
                }
            }
        }
        res_final=tmp;
    }

    // write transforms.txt
    {
        std::ofstream f("/home/harshit/Desktop/CG/Team21_2022209_2022105/HyperSpec/transforms.txt");
        if(!f.is_open()) throw std::runtime_error("Cannot open transforms.txt");

        // The python code loops over B and nframes:
        for(int i=0;i<B;i++){
            std::string s;
            for(int j=0;j<nframes;j++){
                for(int k=0;k<12;k++){
                    std::string val = std::to_string(res_final(i*nframes+j,k));
                    if(!val.empty() && val[0]=='n'){
                        // val=val[11:-1]
                        if(val.size()>12)
                            val=val.substr(11,val.size()-11-1);
                    }
                    s+= val+" ";
                }
            }
            if(!s.empty() && s.back()==' ') s.pop_back();
            f<<s<<"\n";
        }
    }

    // write weights.txt
    {
        std::ofstream f("/home/harshit/Desktop/CG/Team21_2022209_2022105/HyperSpec/weights.txt");
        if(!f.is_open()) throw std::runtime_error("Cannot open weights.txt");
        // weights shape: let's say (B,m)
        // Actually from python:
        // print(weights.shape)
        // m,n = weights.shape[0], weights.shape[1]
        // In python: (m,n)=weights.shape. writes column by column i in range(n): ...
        // code: for i in range(n): ...
        // s += str(j)+" "
        // end line

        // In python code:
        // m, n = weights.shape
        // for i in range(n):
        //   for j in weights[:,i]:
        // means i indexes columns, j indexes rows => So n is columns, m is rows
        // but we have weights(rows,cols)=B x ?

        int m=(int)weights.rows();
        int n=(int)weights.cols();
        // The python code does: for i in range(n):  # columns
        //    for j in weights[:,i]:
        // So we print column-wise again:
        for(int i=0;i<n;i++){
            std::string s;
            for(int r=0;r<m;r++){
                double val=weights(r,i);
                std::string vstr=std::to_string(val);
                if(!vstr.empty() && vstr[0]=='n'){
                    if(vstr.size()>12)
                        vstr=vstr.substr(11,vstr.size()-11-1);
                }
                s+=vstr+" ";
            }
            if(!s.empty() && s.back()==' ') s.pop_back();
            s+="\n";
            f<<s;
        }
    }
}

void write_OBJ(const std::string &path,
                      const std::vector<std::array<double,3>> &vs,
                      const std::vector<std::array<int,3>> &fs) {
    std::ofstream file(path);
    if(!file.is_open()) throw std::runtime_error("Cannot open:"+path);
    for(auto &v: vs) {
        file<<"v "<<v[0]<<" "<<v[1]<<" "<<v[2]<<"\n";
    }
    for(auto &tri: fs) {
        // faces are 0-based in python, 1-based in OBJ
        file<<"f "<<tri[0]+1<<" "<<tri[1]+1<<" "<<tri[2]+1<<"\n";
    }
}

// int main(int argc, char* argv[]){
//     if(argc!=2) {
//         std::cerr<<"Usage: "<<argv[0]<<" path/to/poses.txt\n";
//         return -1;
//     }

//     auto M = load_poses(argv[1]);
//     std::cout<<"poses "<< M.size() <<" x "<< (M.size()>0?M[0].size():0) <<" x 3\n";
//     // print a few values:
//     for(size_t i=0;i<M.size();i++){
//         for(size_t j=0;j<M[i].size();j++){
//             std::cout<<"("<<M[i][j][0]<<","<<M[i][j][1]<<","<<M[i][j][2]<<") ";
//         }
//         std::cout<<"\n";
//     }

//     return 0;
// }