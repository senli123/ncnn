#include "pnnx_graph_parse.h"
int main(int argc, char** argv);
namespace pnnx_graph {

bool PnnxGraph::getNvpPnnxModel(const std::string& pt_path, const std::string& input_shape, const std::string& custom_op_path,
                                const std::string& custom_op_py, const std::string& start_nodes, const std::string& end_nodes)

{
    int argc;
    char** argv;
    if (custom_op_path != "None" && custom_op_py != "None")
    {
        argc = 7;
    }
    else if (custom_op_path != "None" && custom_op_py == "None")
    {
        argc = 6;
    }
    else if (custom_op_path == "None" && custom_op_py == "None")
    {
        argc = 5;
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

    if (custom_op_path != "None")
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

    //insert start nodes
    std::string stard_nodes_info = "start_nodes=" + start_nodes;
    argv[argc - 2] = new char[stard_nodes_info.size() + 1];
    std::strcpy( argv[argc - 2], stard_nodes_info.c_str());

    //insert end nodes
    std::string end_nodes_info = "end_nodes=" + end_nodes;
    argv[argc - 1] = new char[end_nodes_info.size() + 1];
    std::strcpy( argv[argc - 1], end_nodes_info.c_str());

    int result = main(argc, argv);
   
    for (int i = 0; i < argc; ++i)
    {
        delete[] argv[i];
    }

    delete[] argv;

    if (result != -1)
    {
        return true;
    }
    return false;
}

bool PnnxGraph::loadModel(const std::string& param_path, const std::string& bin_path, const std::string& key)
{
    
    // this->graph_ = std::make_unique<Graph>();
    
    auto it = this->graph_map_.find(key);  
    if (it != this->graph_map_.end()) 
    {  
        std::cout << "your input key: " << key << "has been registered"<< std::endl;
        return false;
    }
   
    std::unique_ptr<Graph> graph_;
    graph_ = std::make_unique<Graph>();
   
    int32_t load_result = graph_->load(param_path, bin_path);
    if (load_result != 0)
    {
        std::cout << "Can not find the param path or bin path: " << param_path << " " << bin_path << std::endl;
        return false;
    }
    
    std::cout << "123" << bin_path << std::endl;
    //parse all operator
    std::vector<Operator> operators_;
    std::vector<Operand> operands_;
    std::vector<Operator> input_ops_;
    std::vector<Operator> output_ops_;
    std::vector<Operator*> operators = graph_->ops;

    if (operators.empty())
    {
        std::cout << "Can not read the layers' define" << std::endl;
        return false;
    }
    for (Operator* op : operators)
    {
        if (!op)
        {
            std::cout << "Meet the empty node" << std::endl;
            continue;
        }
        else
        {
            std::map<std::string, Attribute>& attrs = op->attrs;
            for (auto iter = attrs.begin(); iter != attrs.end(); ++iter)
            {
                // std::cout << iter->first << std::endl;
                Attribute& attr = iter->second;
                std::vector<char> data1 = attr.data;
                // py::vector<char> vec_data = py::vector<char>(data1.size(), data1.data());
                attr.b_data = py::bytes(data1.data(), data1.size());
            }
            operators_.push_back(*op);
            if (op->inputs.empty())
            {
                input_ops_.push_back(*op);
            }
            if (op->outputs.empty())
            {
                output_ops_.push_back(*op);
            }
        }
    }
    //parse all operand
    std::vector<Operand*> operands = graph_->operands;

    if (operands.empty())
    {
        std::cout << "Can not read the blob define" << std::endl;
        return false;
    }
    for (Operand* blob : operands)
    {
        if (!blob)
        {
            std::cout << "Meet the empty blob" << std::endl;
            continue;
        }
        else
        {
            operands_.push_back(*blob);
        }
    }

    this->operators_map_[key] = operators_;
    this->operands_map_[key] = operands_;
    this->input_ops_map_[key] = input_ops_;
    this->output_ops_map_[key] = output_ops_;
    this->graph_map_[key] = std::move(graph_); 
    return true;
}

std::vector<Operator> PnnxGraph::getOperators(const std::string& key) const
{
    auto it = this->operators_map_.find(key); 
    std::vector<Operator> operators; 
    if (it != this->operators_map_.end()) 
    {  
        operators = this->operators_map_.at(key);
    }
    else
    {
        std::cout << "your input key: " << key << "has not register"<< std::endl;
        
    }
    return operators;
   
}

std::vector<Operand> PnnxGraph::getOperands(const std::string& key) const
{
   
    auto it = this->operands_map_.find(key);
    std::vector<Operand> operands;  
    if (it != this->operands_map_.end()) 
    {  
        operands = this->operands_map_.at(key);
    }
    else
    {
        std::cout << "your input key: " << key << "has not register"<< std::endl;
        
    }
    return operands;
}

std::vector<Operator> PnnxGraph::getInputOps(const std::string& key) const
{
    
    auto it = this->input_ops_map_.find(key);
    std::vector<Operator> operators;  
    if (it != this->input_ops_map_.end()) 
    {  
        operators = this->input_ops_map_.at(key);
    }
    else
    {
        std::cout << "your input key: " << key << "has not register"<< std::endl;
       
    }
    return operators;
}

std::vector<Operator> PnnxGraph::getOutputOps(const std::string& key) const
{
    auto it = this->output_ops_map_.find(key);
    std::vector<Operator> operators;  
    
    if (it != this->output_ops_map_.end()) 
    {  
        operators = this->output_ops_map_.at(key);
    }
    else
    {
        std::cout << "your input key: " << key << "has not register"<< std::endl;
       
    }
    return operators;
}


bool PnnxGraph::saveModel(const std::string& parampath, const std::vector<Operator>& operators, const std::vector<Operand>& operands, const std::string& key)
{   

    auto it = graph_map_.find(key);  
    if (it != graph_map_.end())
    {  
        // int32_t save_result = this->graph_->save_param(parampath, operators, operands);
        Graph* graphPtr = graph_map_[key].get(); 
        int32_t save_result = graphPtr->save_param(parampath, operators, operands);
        if (save_result != 0)
        {
            std::cout << "Can not save params to param path: " << parampath << std::endl;
            return false;
        }
        return true; 
    }else
    {
        std::cout << "Please input a src model" << std::endl;
    }  
    return false; 
    
}

} // namespace pnnx_graph
