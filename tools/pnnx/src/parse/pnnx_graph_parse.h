#pragma once
#include <string>
#include <memory>
#include <iostream>
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
     * @return true
     * @return false
     */
    bool getNvpPnnxModel(const std::string& pt_path, const std::string& input_shape, const std::string& custom_op_path, const std::string& custom_op_py);
    /**
     * @brief load pnnx graph
     *
     * @param param_path pnnx.param path
     * @param bin_path  pnnx.bin path
     * @return true
     * @return false
     */
    bool loadModel(const std::string& param_path, const std::string& bin_path);

    /**
     * @brief Get the Operator object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operator>>
     */
    std::vector<Operator> getOperators() const;
    /**
     * @brief Get the Operands object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operand>>
     */
    std::vector<Operand> getOperands() const;

    /**
     * @brief Get the Input Ops object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operator>>
     */

    std::vector<Operator> getInputOps() const;

    /**
     * @brief Get the Output Ops object
     *
     * @return std::vector<std::shared_ptr<pnnx::Operator>>
     */
    std::vector<Operator> getOutputOps() const;

private:
    /// @brief load pnnx graph
    std::unique_ptr<Graph> graph_;
    /// @brief  all operator
    std::vector<Operator> operators_;
    /// @brief  all operand
    std::vector<Operand> operands_;
    /// @brief  all input operator
    std::vector<Operator> input_ops_;
    /// @brief  all output operator
    std::vector<Operator> output_ops_;
};
} // namespace pnnx_graph