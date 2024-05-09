#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "parse/pnnx_graph_parse.h"
// #include <torch/extension.h>
#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define MYLIBRARY_VERSION "dev.1.0.5.20240508"
using namespace pnnx_graph;
using namespace pnnx_ir;
namespace py = pybind11;
// class PyParameter:public pnnx::Parameter
// {};
// class PyAttribute:public pnnx::Attribute
// {};
// class PyOperator:public pnnx::Operator
// {};
// class PyOperand:public pnnx::Operand
// {};
PYBIND11_MODULE(ptx, m)
{
    m.doc() = R"pbdoc(
        nvppnnx
        -----------------------

        .. currentmodule:: ptx

        .. autosummary::
           :toctree: _generate

          
    )pbdoc";
    //Parameter class
    py::class_<Parameter>(m,"Parameter")
    .def(py::init<>())
    .def_readwrite("type", &Parameter::type)
    .def_readwrite("b", &Parameter::b)
    .def_readwrite("i", &Parameter::i)
    .def_readwrite("f", &Parameter::f)
    .def_readwrite("s", &Parameter::s)
    .def_readwrite("a_i", &Parameter::ai)
    .def_readwrite("a_f", &Parameter::af)
    .def_readwrite("a_s", &Parameter::as);
    //Attribute class
    py::class_<Attribute>(m, "Attribute")
    .def(py::init<>())
    .def_readwrite("type", &Attribute::type)
    .def_readwrite("shape", &Attribute::shape)
    .def_readwrite("data", &Attribute::data)
    .def_readwrite("b_data", &Attribute::b_data);
    //Operand class
    py::class_<Operand>(m, "Operand")
    .def(py::init<>())
    .def_readwrite("producer", &Operand::producer)
    .def_readwrite("consumers", &Operand::consumers)
    .def_readwrite("type", &Operand::type)
    .def_readwrite("shape", &Operand::shape)
    .def_readwrite("name", &Operand::name)
    .def_readwrite("params", &Operand::params);

    //Operator class
    py::class_<Operator>(m, "Operator")
    .def(py::init<>())
    .def_readwrite("inputs", &Operator::inputs)
    .def_readwrite("outputs", &Operator::outputs)
    .def_readwrite("type", &Operator::type)
    .def_readwrite("name", &Operator::name)
    .def_readwrite("inputnames", &Operator::inputnames)
    .def_readwrite("params", &Operator::params)
    .def_readwrite("attrs", &Operator::attrs);

    //add PnnxGraph class
    py::class_<PnnxGraph>(m, "PnnxGraph")
    .def(py::init<>())
    .def("getNvpPnnxModel", &PnnxGraph::getNvpPnnxModel)
    .def("loadModel", &PnnxGraph::loadModel)
    .def("getOperators", (std::vector<Operator>(PnnxGraph::*)()) & PnnxGraph::getOperators)
    .def("getOperands", &PnnxGraph::getOperands, py::return_value_policy::reference_internal)
    .def("getInputOps", &PnnxGraph::getInputOps, py::return_value_policy::reference_internal)
    .def("getOutputOps", &PnnxGraph::getOutputOps, py::return_value_policy::reference_internal);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = MYLIBRARY_VERSION;
#endif
}
