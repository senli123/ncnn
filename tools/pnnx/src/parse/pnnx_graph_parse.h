#pragma once
#include <string>
#include <memory>
#include <iostream>
#include <unordered_map> 
#include "pnnx_ir_parse.h"
using namespace pnnx_ir;
namespace pnnx_graph {

class PnnxGraph
{
public:

    /**
     * @brief Get the Nvp Pnnx Model object
     *
     * @param pt_path torchscript path
     * @param input_shape input shape of tensor
     * @param custom_op_path the path of define custom op
     * @param custom_op_py the py path of define custom op
     * @param start_nodes the list of start nodes
     * @param end_nodes the list of end nodes
     * @return true
     * @return false
     */
    bool getNvpPnnxModel(const std::string& pt_path, \
    const std::string& input_shape, \
    const std::string& custom_op_path, \
    const std::string& custom_op_py,
    const std::string& start_nodes = "",
    const std::string& end_nodes = "");
    
    /**
     * @brief load pnnx graph
     *
     * @param param_path pnnx.param path
     * @param bin_path  pnnx.bin path
     * @param key  model name
     * @return true
     * @return false
     */
    bool loadModel(const std::string& param_path, const std::string& bin_path, const std::string& key);

    /**
     * @brief 
     * 
     * @param parampath pnnx.param path
     * @param operators input ops
     * @param operands input operands
     * @return true 
     * @return false 
     */
    bool saveModel(const std::string& parampath, const std::vector<Operator>& operators, const std::vector<Operand>& operands, const std::string& key);
    
    /**
     * @brief Get the Operator object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operator>>
     */
    std::vector<Operator> getOperators(const std::string& key) const;
    /**
     * @brief Get the Operands object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operand>>
     */
    std::vector<Operand> getOperands(const std::string& key) const;

    /**
     * @brief Get the Input Ops object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operator>>
     */

    std::vector<Operator> getInputOps(const std::string& key) const;

    /**
     * @brief Get the Output Ops object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operator>>
     */
    std::vector<Operator> getOutputOps(const std::string& key) const;


     

private:
    /// @brief load pnnx graph
    // std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, std::unique_ptr<Graph>> graph_map_;
    /// @brief  all operator
    std::unordered_map<std::string, std::vector<Operator>> operators_map_;
    /// @brief  all operand
    std::unordered_map<std::string, std::vector<Operand>> operands_map_;
    /// @brief  all input operator
    std::unordered_map<std::string, std::vector<Operator>> input_ops_map_;
    /// @brief  all output operator
    std::unordered_map<std::string, std::vector<Operator>> output_ops_map_;
};
} // namespace pnnx_graph