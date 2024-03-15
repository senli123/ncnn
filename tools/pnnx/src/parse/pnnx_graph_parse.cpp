#include "pnnx_graph_parse.h"
int main(int argc, char** argv);
namespace pnnx_graph{

bool PnnxGraph::getNvpPnnxModel(const std::string& pt_path, const std::string& input_shape, const std::string& custom_op_path, \
const std::string& custom_op_py)
{
    
    int argc;
    char** argv;
    if(custom_op_path != "None" && custom_op_py != "None")
    {
        argc = 5;

    }
    else if(custom_op_path != "None" && custom_op_py == "None")
    {
        argc = 4;
         
    }
    else if(custom_op_path == "None" && custom_op_py == "None")
    {
        argc = 3;

    }
    argv = new char*[argc];

    argv[0] = new char[1];  
    argv[0][0] = '\0';  

    //insert pt_path
    argv[1] = new char[pt_path.size() + 1];
    std::strcpy(argv[1], pt_path.c_str());

    //insert input_shape
    std::string input_shape_info = "inputshape=" + input_shape; 
    argv[2] = new char[input_shape_info.size() + 1];
    std::strcpy(argv[2], input_shape_info.c_str());

    if(custom_op_path != "None" )
    {
       //insert custom_op
        std::string custom_op_info = "customop=" + custom_op_path; 
        argv[3] = new char[custom_op_info.size() + 1];
        std::strcpy(argv[3], custom_op_info.c_str());
         
    }

    if (custom_op_py != "None")
    {
        //insert custom_op_py
        std::string custom_op_py_info = "customop_infer_py=" + custom_op_py; 
        argv[4] = new char[custom_op_py_info.size() + 1];
        std::strcpy(argv[4], custom_op_py_info.c_str());
        
    }
    
    int result = main(argc, argv); 

    for (int i = 0; i < argc; ++i) {  
        delete[] argv[i];  
    }  

    delete[] argv;  
  
    if (result != -1)
    {
        return true;
    } 
    return false;   
}  

bool PnnxGraph::loadModel(const std::string& param_path, const std::string& bin_path)
{
    this->graph_ = std::make_unique<Graph>();
    int32_t load_result = this->graph_->load(param_path, bin_path);
    if (load_result != 0) {
        std::cout << "Can not find the param path or bin path: " << param_path << " " << bin_path <<std::endl;
        return false;
    }
    //parse all operator
    this->operators_.clear();
    this->input_ops_.clear();
    this->output_ops_.clear();
    std::vector<Operator*> operators = this->graph_->ops;
    
    if (operators.empty()) {
        std::cout << "Can not read the layers' define"<<std::endl;
        return false;
    }
    for (Operator* op : operators) {
        if (!op) {
            std::cout << "Meet the empty node" << std::endl;
            continue;
        } else {
            std::map<std::string, Attribute>& attrs = op->attrs;
            for (auto iter = attrs.begin(); iter != attrs.end(); ++iter) {
                // std::cout << iter->first << std::endl;
                Attribute& attr = iter->second;
                std::vector<char> data1 = attr.data;
                // py::vector<char> vec_data = py::vector<char>(data1.size(), data1.data()); 
                attr.b_data = py::bytes(data1.data(),data1.size());
            }
            this->operators_.push_back(*op);
            if(op->inputs.empty()){
                this->input_ops_.push_back(*op);
            }
            if(op->outputs.empty()){
                this->output_ops_.push_back(*op);
            }

        }
    }
    //parse all operand
    this->operands_.clear();
    std::vector<Operand*> operands = this->graph_->operands;
    
    if (operands.empty()) {
        std::cout << "Can not read the blob define"<<std::endl;
        return false;
    }
    for (Operand* blob : operands) {
        if (!blob) {
            std::cout << "Meet the empty blob" << std::endl;
            continue;
        } else {
            this->operands_.push_back(*blob);  

        }
    }
    return true;
}


 std::vector<Operator> PnnxGraph::getOperators() const
{

    return this->operators_;
}

 std::vector<Operand> PnnxGraph::getOperands() const
{
    return this->operands_;
}

 std::vector<Operator> PnnxGraph::getInputOps() const
{
    return this->input_ops_;
}

 std::vector<Operator> PnnxGraph::getOutputOps() const
{
    return this->output_ops_;
}


}
