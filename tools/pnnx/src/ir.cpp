// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "ir.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <regex>
#include <cstdio>
#include <iostream>
#include "storezip.h"
#include "utils.h"
#include <list>
namespace pnnx {

static bool type_is_integer(int type)
{
    if (type == 1) return false;
    if (type == 2) return false;
    if (type == 3) return false;
    if (type == 4) return true;
    if (type == 5) return true;
    if (type == 6) return true;
    if (type == 7) return true;
    if (type == 8) return true;
    if (type == 9) return true;
    if (type == 10) return false;
    if (type == 11) return false;
    if (type == 12) return false;
    return false;
}

static const char* type_to_string(int type)
{
    if (type == 1) return "f32";
    if (type == 2) return "f64";
    if (type == 3) return "f16";
    if (type == 4) return "i32";
    if (type == 5) return "i64";
    if (type == 6) return "i16";
    if (type == 7) return "i8";
    if (type == 8) return "u8";
    if (type == 9) return "bool";
    if (type == 10) return "c64";
    if (type == 11) return "c128";
    if (type == 12) return "c32";
    return "null";
}

static const char* type_to_numpy_string(int type)
{
    if (type == 1) return "float32";
    if (type == 2) return "float64";
    if (type == 3) return "float16";
    if (type == 4) return "int32";
    if (type == 5) return "int64";
    if (type == 6) return "int16";
    if (type == 7) return "int8";
    if (type == 8) return "uint8";
    if (type == 9) return "bool8";
    if (type == 10) return "csingle";
    if (type == 11) return "cdouble";
    if (type == 12) return "chalf";
    return "null";
}

static const char* type_to_dtype_string(int type)
{
    if (type == 1) return "torch.float";
    if (type == 2) return "torch.double";
    if (type == 3) return "torch.half";
    if (type == 4) return "torch.int";
    if (type == 5) return "torch.long";
    if (type == 6) return "torch.short";
    if (type == 7) return "torch.int8";
    if (type == 8) return "torch.uint8";
    if (type == 9) return "torch.bool";
    if (type == 10) return "torch.complex64";
    if (type == 11) return "torch.complex128";
    if (type == 12) return "torch.complex32";
    return "null";
}

static size_t type_to_elemsize(int type)
{
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;
    return 0; // null
}

static int string_to_type(const char* s)
{
    if (strcmp(s, "f32") == 0) return 1;
    if (strcmp(s, "f64") == 0) return 2;
    if (strcmp(s, "f16") == 0) return 3;
    if (strcmp(s, "i32") == 0) return 4;
    if (strcmp(s, "i64") == 0) return 5;
    if (strcmp(s, "i16") == 0) return 6;
    if (strcmp(s, "i8") == 0) return 7;
    if (strcmp(s, "u8") == 0) return 8;
    if (strcmp(s, "bool") == 0) return 9;
    if (strcmp(s, "c64") == 0) return 10;
    if (strcmp(s, "c128") == 0) return 11;
    if (strcmp(s, "c32") == 0) return 12;
    return 0; // null
}

bool operator==(const Parameter& lhs, const Parameter& rhs)
{
    if (lhs.type != rhs.type)
        return false;

    if (lhs.type == 0)
        return true;

    if (lhs.type == 1 && lhs.b == rhs.b)
        return true;

    if (lhs.type == 2 && lhs.i == rhs.i)
        return true;

    if (lhs.type == 3 && lhs.f == rhs.f)
        return true;

    if (lhs.type == 4 && lhs.s == rhs.s)
        return true;

    if (lhs.type == 5 && lhs.ai == rhs.ai)
        return true;

    if (lhs.type == 6 && lhs.af == rhs.af)
        return true;

    if (lhs.type == 7 && lhs.as == rhs.as)
        return true;

    if (lhs.type == 10 && lhs.c == rhs.c)
        return true;

    if (lhs.type == 11 && lhs.ac == rhs.ac)
        return true;

    return false;
}

Attribute::Attribute(const std::initializer_list<int>& _shape, const std::vector<float>& t)
{
    type = 1;
    shape = _shape;

    if (shape.size() > 0)
    {
        data.resize(elemcount() * type_to_elemsize(type));
        memcpy((void*)data.data(), (const void*)t.data(), data.size());
    }
}

size_t Attribute::elemsize() const
{
    return type_to_elemsize(type);
}

int Attribute::elemcount() const
{
    if (shape.empty())
        return 0;

    int size = shape[0];
    for (size_t i = 1; i < shape.size(); i++)
    {
        size *= shape[i];
    }

    return size;
}

std::vector<float> Attribute::get_float32_data() const
{
    std::vector<float> v(elemcount());

    if (type == 1)
    {
        memcpy((void*)v.data(), (const void*)data.data(), data.size());
    }
    else if (type == 2)
    {
        // f64
        const double* p = (const double*)data.data();
        for (size_t i = 0; i < v.size(); i++)
        {
            v[i] = float(p[i]);
        }
    }
    else if (type == 3)
    {
        // f16
        const unsigned short* p = (const unsigned short*)data.data();
        for (size_t i = 0; i < v.size(); i++)
        {
            v[i] = float16_to_float32(p[i]);
        }
    }
    else
    {
        fprintf(stderr, "cannot convert type %d to float32 data\n", type);
    }

    return v;
}

void Attribute::set_float32_data(const std::vector<float>& newdata)
{
    data.resize(newdata.size() * elemsize());

    if (type == 1)
    {
        memcpy((void*)data.data(), (const void*)newdata.data(), data.size());
    }
    else if (type == 2)
    {
        // f64
        double* p = (double*)data.data();
        for (size_t i = 0; i < newdata.size(); i++)
        {
            p[i] = newdata[i];
        }
    }
    else if (type == 3)
    {
        // f16
        unsigned short* p = (unsigned short*)data.data();
        for (size_t i = 0; i < newdata.size(); i++)
        {
            p[i] = float32_to_float16(newdata[i]);
        }
    }
    else
    {
        fprintf(stderr, "cannot convert float32 data to type %d\n", type);
    }
}

bool operator==(const Attribute& lhs, const Attribute& rhs)
{
    if (lhs.type != rhs.type)
        return false;

    if (lhs.type == 0)
        return true;

    if (lhs.shape != rhs.shape)
        return false;

    if (lhs.data != rhs.data)
        return false;

    return true;
}

Attribute operator+(const Attribute& a, const Attribute& b)
{
    Attribute c;

    if (a.type != b.type)
    {
        fprintf(stderr, "concat attribute type mismatch\n");
        return c;
    }

    if (a.shape.size() != b.shape.size())
    {
        fprintf(stderr, "concat attribute shape rank mismatch\n");
        return c;
    }

    for (int i = 1; i < (int)a.shape.size(); i++)
    {
        if (a.shape[i] != b.shape[i])
        {
            fprintf(stderr, "concat attribute shape mismatch\n");
            return c;
        }
    }

    c.type = a.type;
    c.shape = a.shape;
    c.shape[0] += b.shape[0]; // concat the first dim

    c.data.resize(a.data.size() + b.data.size());
    memcpy(c.data.data(), a.data.data(), a.data.size());
    memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());

    return c;
}

Parameter Parameter::parse_from_string(const std::string& value)
{
    if (value.find('%') != std::string::npos)
    {
        Parameter p;
        p.type = 4;
        p.s = value;
        return p;
    }

    Parameter p;
    p.type = 0;

    if (value == "None" || value == "()" || value == "[]")
    {
        return p;
    }

    if (value == "True" || value == "False")
    {
        // bool
        p.type = 1;
        p.b = value == "True";
        return p;
    }

    if (value[0] == '(' || value[0] == '[')
    {
        // list
        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);

        while (!lcss.eof())
        {
            std::string elem;
            std::getline(lcss, elem, ',');

            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9')))
            {
                // string
                p.type = 7;
                p.as.push_back(elem);
            }
            else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos)
            {
                // float
                p.type = 6;
                p.af.push_back(std::stof(elem));
            }
            else
            {
                // integer
                p.type = 5;
                p.ai.push_back(std::stoi(elem));
            }
        }
        return p;
    }

    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9')))
    {
        // string
        p.type = 4;
        p.s = value;
        return p;
    }

    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos)
    {
        // float
        p.type = 3;
        p.f = std::stof(value);
        return p;
    }

    // integer
    p.type = 2;
    p.i = std::stoi(value);
    return p;
}

std::string Parameter::encode_to_string(const Parameter& param)
{
    if (param.type == 0)
    {
        return std::string("None");
    }
    if (param.type == 1)
    {
        if (param.b)
            return std::string("True");
        else
            return std::string("False");
    }
    if (param.type == 2)
    {
        return std::to_string(param.i);
    }
    if (param.type == 3)
    {
        char buf[64];
        sprintf(buf, "%e", param.f);
        return std::string(buf);
    }
    if (param.type == 4)
    {
        return param.s;
    }
    if (param.type == 5)
    {
        std::string s("(");
        for (size_t i = 0; i < param.ai.size(); i++)
        {
            s += std::to_string(param.ai[i]);
            if (i + 1 != param.ai.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }
    if (param.type == 6)
    {
        std::string s("(");
        for (size_t i = 0; i < param.af.size(); i++)
        {
            char buf[64];
            sprintf(buf, "%e", param.af[i]);
            s += std::string(buf);
            if (i + 1 != param.af.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }
    if (param.type == 7)
    {
        std::string s("(");
        for (size_t i = 0; i < param.as.size(); i++)
        {
            s += param.as[i];
            if (i + 1 != param.as.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }
    if (param.type == 10)
    {
        char buf[128];
        sprintf(buf, "%e+%ej", param.c.real(), param.c.imag());
        return std::string(buf);
    }
    if (param.type == 11)
    {
        std::string s("(");
        for (size_t i = 0; i < param.ac.size(); i++)
        {
            char buf[128];
            sprintf(buf, "%e+%ej", param.ac[i].real(), param.ac[i].imag());
            s += std::string(buf);
            if (i + 1 != param.ac.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }

    fprintf(stderr, "unknown parameter type %d\n", param.type);
    return std::string();
}

bool Operator::has_param(const std::string& key) const
{
    return params.find(key) != params.end();
}

bool Operator::has_attr(const std::string& key) const
{
    return attrs.find(key) != attrs.end();
}

bool Operator::has_input(const std::string& key) const
{
    return std::find(inputnames.begin(), inputnames.end(), key) != inputnames.end();
}

Operand* Operator::named_input(const std::string& key)
{
    for (size_t i = 0; i < inputnames.size(); i++)
    {
        if (inputnames[i] == key)
            return inputs[i];
    }

    return 0;
}

const Operand* Operator::named_input(const std::string& key) const
{
    for (size_t i = 0; i < inputnames.size(); i++)
    {
        if (inputnames[i] == key)
            return inputs[i];
    }

    return 0;
}

Graph::Graph()
{
}

Graph::~Graph()
{
    for (auto x : ops)
        delete x;

    for (auto x : operands)
        delete x;

    ops.clear();
    operands.clear();
}

Graph::Graph(const Graph& /*rhs*/)
{
}

Graph& Graph::operator=(const Graph& /*rhs*/)
{
    return *this;
}

static void load_parameter(Operator* op, const std::string& key, const std::string& value)
{
    op->params[key] = Parameter::parse_from_string(value);
}

static void load_input_key(Operator* op, const std::string& key, const std::string& value)
{
    op->inputnames.resize(op->inputs.size());

    for (size_t i = 0; i < op->inputs.size(); i++)
    {
        const Operand* oprand = op->inputs[i];
        if (oprand->name == value)
        {
            op->inputnames[i] = key;
            break;
        }
    }
}

static void load_shape(Operator* op, const std::string& key, const std::string& value)
{
    Operand* operand = 0;
    for (auto r : op->inputs)
    {
        if (r->name == key)
        {
            operand = r;
            break;
        }
    }

    if (!operand)
    {
        for (auto r : op->outputs)
        {
            if (r->name == key)
            {
                operand = r;
                break;
            }
        }
    }

    if (!operand)
    {
        fprintf(stderr, "no such operand %s for operator %s\n", key.c_str(), op->name.c_str());
        return;
    }

    // type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    operand->type = string_to_type(typestr.c_str());

    // shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);

    operand->shape.clear();
    while (!lcss.eof())
    {
        std::string elem;
        std::getline(lcss, elem, ',');

        if (elem == "?")
        {
            operand->shape.push_back(-1);
        }
        else if (elem[0] == '%')
        {
            // encode %abc as symbolic tag
            operand->shape.push_back(-233);
            int index = operand->shape.size() - 1;
            std::string key = elem.substr(1);
            operand->params[std::string("__shape_") + std::to_string(index)] = key;
        }
        else
        {
            int i = std::stoi(elem);
            operand->shape.push_back(i);
        }
    }
}

static void load_attribute(Operator* op, const std::string& key, const std::string& value, StoreZipReader& szr)
{
    Attribute& a = op->attrs[key];

    // type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    a.type = string_to_type(typestr.c_str());

    if (a.type == 0)
        return;

    // shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);

    a.shape.clear();
    while (!lcss.eof())
    {
        std::string elem;
        std::getline(lcss, elem, ',');

        int i = std::stoi(elem);
        a.shape.push_back(i);
    }

    if (a.shape.empty())
        return;

    // data
    size_t size = 1;
    for (int i : a.shape)
    {
        size *= i;
    }

    size_t bytesize = size * type_to_elemsize(a.type);

    std::string filename = op->name + "." + key;

    size_t filesize = szr.get_file_size(filename);

    if (filesize == 0)
    {
        // no such file
        return;
    }

    if (filesize != bytesize)
    {
        fprintf(stderr, "file size not match expect %lu but got %lu\n", bytesize, filesize);
    }

    a.data.resize(bytesize);
    szr.read_file(filename, (char*)a.data.data());
}

int Graph::load(const std::string& parampath, const std::string& binpath)
{
    std::ifstream is(parampath, std::ios::in | std::ios::binary);
    if (!is.good())
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    StoreZipReader szr;
    if (szr.open(binpath) != 0)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int magic = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> magic;
    }

    int operator_count = 0;
    int operand_count = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> operator_count >> operand_count;
    }

    for (int i = 0; i < operator_count; i++)
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int input_count = 0;
        int output_count = 0;

        iss >> type >> name >> input_count >> output_count;

        Operator* op = new_operator(type, name);

        for (int j = 0; j < input_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = get_operand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < output_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = new_operand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
        }

        // key=value
        while (!iss.eof())
        {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);

            if (key[0] == '@')
            {
                // attribute
                load_attribute(op, key.substr(1), value, szr);
            }
            else if (key[0] == '$')
            {
                // operand input key
                load_input_key(op, key.substr(1), value);
            }
            else if (key[0] == '#')
            {
                // operand shape
                load_shape(op, key.substr(1), value);
            }
            else
            {
                // parameter
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

int Graph::save(const std::string& parampath, const std::string& binpath)
{
    FILE* paramfp = fopen(parampath.c_str(), "wb");
    if (!paramfp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath.c_str());
        return -1;
    }

    StoreZipWriter szw;
    if (szw.open(binpath) != 0)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    // magic
    fprintf(paramfp, "7767517\n");

    // op count and oprand count
    fprintf(paramfp, "%d %d\n", (int)ops.size(), (int)operands.size());

    for (const Operator* op : ops)
    {
        fprintf(paramfp, "%-24s %-24s %d %d", op->type.c_str(), op->name.c_str(), (int)op->inputs.size(), (int)op->outputs.size());

        for (const Operand* oprand : op->inputs)
        {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const Operand* oprand : op->outputs)
        {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const auto& it : op->params)
        {
            fprintf(paramfp, " %s=", it.first.c_str());

            const Parameter& param = it.second;
            std::string s = Parameter::encode_to_string(param);
            fprintf(paramfp, "%s", s.c_str());
        }

        for (const auto& it : op->attrs)
        {
            fprintf(paramfp, " @%s=", it.first.c_str());

            const Attribute& attr = it.second;
            fprintf(paramfp, "(");
            for (int i = 0; i < (int)attr.shape.size() - 1; i++)
            {
                fprintf(paramfp, "%d,", attr.shape[i]);
            }
            if (attr.shape.size() > 0)
                fprintf(paramfp, "%d", attr.shape[attr.shape.size() - 1]);
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(attr.type));

            std::string filename = op->name + "." + it.first;
            szw.write_file(filename, attr.data.data(), attr.data.size());
        }

        if (op->inputnames.size() == op->inputs.size())
        {
            for (size_t i = 0; i < op->inputs.size(); i++)
            {
                if (op->inputnames[i].empty())
                    continue;

                const Operand* oprand = op->inputs[i];
                fprintf(paramfp, " $%s=%s", op->inputnames[i].c_str(), oprand->name.c_str());
            }
        }

        for (const Operand* oprand : op->inputs)
        {
            if (oprand->shape.empty())
                continue;

            fprintf(paramfp, " #%s=", oprand->name.c_str());

            fprintf(paramfp, "(");
            for (int i = 0; i < (int)oprand->shape.size() - 1; i++)
            {
                if (oprand->shape[i] == -1)
                    fprintf(paramfp, "?,");
                else
                    fprintf(paramfp, "%d,", oprand->shape[i]);
            }
            if (oprand->shape.size() > 0)
            {
                if (oprand->shape[oprand->shape.size() - 1] == -1)
                    fprintf(paramfp, "?");
                else
                    fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(oprand->type));
        }

        for (const Operand* oprand : op->outputs)
        {
            if (oprand->shape.empty())
                continue;

            fprintf(paramfp, " #%s=", oprand->name.c_str());

            fprintf(paramfp, "(");
            for (int i = 0; i < (int)oprand->shape.size() - 1; i++)
            {
                if (oprand->shape[i] == -1)
                    fprintf(paramfp, "?,");
                else
                    fprintf(paramfp, "%d,", oprand->shape[i]);
            }
            if (oprand->shape.size() > 0)
            {
                if (oprand->shape[oprand->shape.size() - 1] == -1)
                    fprintf(paramfp, "?");
                else
                    fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(oprand->type));
        }

        fprintf(paramfp, "\n");
    }

    fclose(paramfp);

    return 0;
}

static std::string sanitize_identifier(const std::string& s)
{
    std::string ss = s;
    for (size_t i = 0; i < ss.size(); i++)
    {
        if (ss[i] == '.' || ss[i] == ':' || ss[i] == '/')
            ss[i] = '_';
    }

    return ss;
}

static std::string expand_expression(const Operator* op)
{
    std::string expr = op->params.at("expr").s;

    // split into tokens
    std::vector<std::string> tokens;
    {
        std::string t;
        for (size_t i = 0; i < expr.size(); i++)
        {
            char ch = expr[i];

            if (ch == '[') // list
            {
                t += ch;
                tokens.push_back(t);
                t.clear();
            }
            else if (ch == '(' || ch == ')' || ch == ',' || ch == ']')
            {
                if (!t.empty())
                {
                    tokens.push_back(t);
                    t.clear();
                }
            }
            else
            {
                t += ch;
            }
        }

        if (!t.empty())
        {
            tokens.push_back(t);
        }
    }

    // scan and stack
    std::stack<std::string> exprstack;
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];

        if (t == "size")
        {
            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            std::string r = a + ".size(" + b + ")";
            exprstack.push(r);
        }
        else if (t == "int"
                 || t == "abs"
                 || t == "acos"
                 || t == "acosh"
                 || t == "asin"
                 || t == "asinh"
                 || t == "atan"
                 || t == "atanh"
                 || t == "ceil"
                 || t == "cos"
                 || t == "cosh"
                 || t == "exp"
                 || t == "floor"
                 || t == "log"
                 || t == "log10"
                 || t == "neg"
                 || t == "reciprocal"
                 || t == "round"
                 || t == "rsqrt"
                 || t == "sign"
                 || t == "sin"
                 || t == "sinh"
                 || t == "sqrt"
                 || t == "square"
                 || t == "tan"
                 || t == "tanh"
                 || t == "trunc")
        {
            std::string unaryop;
            if (t == "int") unaryop = "int";
            if (t == "abs") unaryop = "torch.abs";
            if (t == "acos") unaryop = "torch.acos";
            if (t == "acosh") unaryop = "torch.acosh";
            if (t == "asin") unaryop = "torch.asin";
            if (t == "asinh") unaryop = "torch.asinh";
            if (t == "atan") unaryop = "torch.atan";
            if (t == "atanh") unaryop = "torch.atanh";
            if (t == "ceil") unaryop = "torch.ceil";
            if (t == "cos") unaryop = "torch.cos";
            if (t == "cosh") unaryop = "torch.cosh";
            if (t == "exp") unaryop = "torch.exp";
            if (t == "floor") unaryop = "torch.floor";
            if (t == "log") unaryop = "torch.log";
            if (t == "log10") unaryop = "torch.log10";
            if (t == "neg") unaryop = "-";
            if (t == "reciprocal") unaryop = "torch.reciprocal";
            if (t == "round") unaryop = "torch.round";
            if (t == "rsqrt") unaryop = "torch.rsqrt";
            if (t == "sign") unaryop = "torch.sign";
            if (t == "sin") unaryop = "torch.sin";
            if (t == "sinh") unaryop = "torch.sinh";
            if (t == "sqrt") unaryop = "torch.sqrt";
            if (t == "square") unaryop = "torch.square";
            if (t == "tan") unaryop = "torch.tan";
            if (t == "tanh") unaryop = "torch.tanh";
            if (t == "trunc") unaryop = "torch.trunc";

            std::string a = exprstack.top();
            exprstack.pop();

            std::string r = unaryop + "(" + a + ")";
            exprstack.push(r);
        }
        else if (t == "atan2"
                 || t == "fmod"
                 || t == "max"
                 || t == "maximum"
                 || t == "min"
                 || t == "minimum"
                 || t == "pow")
        {
            std::string binaryop;
            if (t == "atan2") binaryop = "torch.atan2";
            if (t == "fmod") binaryop = "torch.fmod";
            if (t == "max") binaryop = "torch.max";
            if (t == "maximum") binaryop = "torch.maximum";
            if (t == "min") binaryop = "torch.min";
            if (t == "minimum") binaryop = "torch.minimum";
            if (t == "pow") binaryop = "torch.pow";

            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            std::string r = binaryop + "(" + a + ", " + b + ")";
            exprstack.push(r);
        }
        else if (t == "add"
                 || t == "sub"
                 || t == "mul"
                 || t == "div"
                 || t == "floor_divide"
                 || t == "remainder"
                 || t == "and"
                 || t == "or"
                 || t == "xor"
                 || t == "lshift"
                 || t == "rshift")
        {
            std::string binaryop;
            if (t == "add") binaryop = "+";
            if (t == "sub") binaryop = "-";
            if (t == "mul") binaryop = "*";
            if (t == "div") binaryop = "/";
            if (t == "floor_divide") binaryop = "//";
            if (t == "remainder") binaryop = "%";
            if (t == "and") binaryop = "&";
            if (t == "or") binaryop = "|";
            if (t == "xor") binaryop = "^";
            if (t == "lshift") binaryop = "<<";
            if (t == "rshift") binaryop = ">>";

            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            std::string r = std::string("(") + a + " " + binaryop + " " + b + ")";
            exprstack.push(r);
        }
        else if (t == "[") // list
        {
            std::vector<std::string> elements;
            while (!exprstack.empty())
            {
                std::string a = exprstack.top();
                exprstack.pop();

                elements.push_back(a);
            }

            std::string r = "[";
            for (int j = 0; j < (int)elements.size() - 1; j++)
            {
                r += elements[j];
                if (j + 1 != (int)elements.size())
                    r += ", ";
            }
            if (!elements.empty())
            {
                r += elements[elements.size() - 1];
            }
            r += "]";

            exprstack.push(r);
        }
        else if (t[0] == '@')
        {
            int input_index = std::stoi(t.substr(1));
            std::string varid = std::string("v_") + sanitize_identifier(op->inputs[input_index]->name);
            exprstack.push(varid);
        }
        else
        {
            // literal
            if (t[t.size() - 1] == 'j')
            {
                // complex
                std::string r = std::string("(") + t + ")";
                exprstack.push(r);
            }
            else
            {
                exprstack.push(t);
            }
        }
    }

    std::string r = exprstack.top();
    exprstack.pop();

    return r;
}

static std::string make_slice_expression(const Operator* op)
{
    // for (size_t j = 0; j < op->inputnames.size(); j++)
    // {
    //     fprintf(stderr, "make_slice_expression %s %s\n", op->inputnames[j].c_str(), op->inputs[j]->name.c_str());
    // }

    std::vector<int> dims;
    if (op->has_param("dims"))
    {
        dims = op->params.at("dims").ai;
    }
    else
    {
        dims.push_back(op->params.at("dim").i);
    }

    std::string pr;
    std::string nr;

    int last_dim = -1;
    const int ndim = (int)dims.size();
    for (int i = 0; i < ndim; i++)
    {
        int dim = dims[i];
        std::string& r = dim < 0 ? nr : pr;

        for (int j = last_dim + 1; j < dim; j++)
        {
            r += ":,";
        }
        last_dim = dim;

        bool is_select = false;
        if (op->has_param("select"))
        {
            int select = op->params.at("select").i;
            if (select != INT_MAX)
            {
                r += std::to_string(select);
                is_select = true;
            }
        }
        if (op->has_param("selects"))
        {
            std::vector<int> selects = op->params.at("selects").ai;
            int select = selects[i];
            if (select != INT_MAX)
            {
                r += std::to_string(select);
                is_select = true;
            }
        }
        if (op->has_input("select"))
        {
            r += std::string("v_") + sanitize_identifier(op->named_input("select")->name);
            is_select = true;
        }
        if (op->has_input("selects"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("selects")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int selecti = std::stoi(index.substr(1));
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[selecti]->name);
                is_select = true;
            }
            else
            {
                int select = std::stoi(index);
                if (select != INT_MAX)
                {
                    r += std::to_string(select);
                    is_select = true;
                }
            }
        }

        if (is_select)
        {
            if (i + 1 != ndim)
                r += ',';
            continue;
        }

        if (op->has_param("start"))
        {
            int start = op->params.at("start").i;
            if (start != 0)
                r += std::to_string(start);
        }
        else if (op->has_param("starts"))
        {
            std::vector<int> starts = op->params.at("starts").ai;
            int start = starts[i];
            if (start != 0)
                r += std::to_string(start);
        }
        else if (op->has_input("start"))
        {
            r += std::string("v_") + sanitize_identifier(op->named_input("start")->name);
        }
        else // if (op->has_input("starts"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("starts")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int starti = std::stoi(index.substr(1));
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[starti]->name);
            }
            else
            {
                int start = std::stoi(index);
                if (start != 0)
                    r += std::to_string(start);
            }
        }

        r += ':';

        if (op->has_param("end"))
        {
            int end = op->params.at("end").i;
            if (end != INT_MAX)
                r += std::to_string(end);
        }
        else if (op->has_param("ends"))
        {
            std::vector<int> ends = op->params.at("ends").ai;
            int end = ends[i];
            if (end != INT_MAX)
                r += std::to_string(end);
        }
        else if (op->has_input("end"))
        {
            r += std::string("v_") + sanitize_identifier(op->named_input("end")->name);
        }
        else // if (op->has_input("ends"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("ends")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int endi = std::stoi(index.substr(1));
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[endi]->name);
            }
            else
            {
                int end = std::stoi(index);
                if (end != INT_MAX)
                    r += std::to_string(end);
            }
        }

        if (op->has_param("step"))
        {
            int step = op->params.at("step").i;
            if (step != 1)
            {
                r += ':';
                r += std::to_string(step);
            }
        }
        else if (op->has_param("steps"))
        {
            std::vector<int> steps = op->params.at("steps").ai;
            int step = steps[i];
            if (step != 1)
            {
                r += ':';
                r += std::to_string(step);
            }
        }
        else if (op->has_input("step"))
        {
            r += ':';
            r += std::string("v_") + sanitize_identifier(op->named_input("step")->name);
        }
        else // if (op->has_input("steps"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("steps")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int stepi = std::stoi(index.substr(1));
                r += ':';
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[stepi]->name);
            }
            else
            {
                int step = std::stoi(index);
                if (step != 1)
                {
                    r += ':';
                    r += std::to_string(step);
                }
            }
        }

        if (i + 1 != ndim)
            r += ',';
    }

    if (!pr.empty() && !nr.empty())
        return pr + "...," + nr;

    if (pr.empty() && !nr.empty())
        return std::string("...,") + nr;

    return pr + nr;
}

static std::string make_index_expression(const Operator* op)
{
    fprintf(stderr, "make_index_expression %s\n", op->name.c_str());

    std::string index_expr = op->params.at("expr").s;

    // strip out-most [ ] pair
    // index_expr = index_expr.substr(1, index_expr.size() - 2);

    // // None,None,   ->   ...,
    // bool leading_none = false;
    // while (index_expr.substr(0, 5) == "None,")
    // {
    //     leading_none = true;
    //     index_expr = index_expr.substr(5);
    // }
    // if (leading_none)
    // {
    //     index_expr = "...," + index_expr;
    // }

    // return index_expr;
    std::vector<int> shape = op->inputs.at(0)->shape;
    std::string out_index_expr = "";
    index_expr = index_expr.substr(1, index_expr.size() - 2);
    int indices_index = 0;
    while (index_expr.substr(0, 5) == "None,")
    {
      
        index_expr = index_expr.substr(5);
        indices_index++;
    }
    size_t pos = 0;  
    if ((pos = index_expr.find("@")) != std::string::npos) {  
        index_expr.replace(pos, 1, "v_");  
    }
    for(int i = 0; i < shape.size(); i++)
    {
        if ( i == indices_index)
        {
            out_index_expr = out_index_expr + index_expr;

        }else
        {
            out_index_expr =  out_index_expr + ":";
            
        }
        if ( i != shape.size() - 1)
        {
             out_index_expr =  out_index_expr + ",";
        }
    }
    return out_index_expr;
}

int Graph::python(const std::string& pypath, const std::string& pnnxbinpath)
{
    FILE* pyfp = fopen(pypath.c_str(), "wb");
    if (!pyfp)
    {
        fprintf(stderr, "fopen %s failed\n", pypath.c_str());
        return -1;
    }

    fprintf(pyfp, "import os\n");
    fprintf(pyfp, "import numpy as np\n");
    fprintf(pyfp, "import tempfile, zipfile\n");
    fprintf(pyfp, "import torch\n");
    fprintf(pyfp, "import torch.nn as nn\n");
    fprintf(pyfp, "import torch.nn.functional as F\n");
    fprintf(pyfp, "try:\n");
    fprintf(pyfp, "    import torchvision\n");
    fprintf(pyfp, "except:\n");
    fprintf(pyfp, "    pass\n");

    fprintf(pyfp, "\n");

    fprintf(pyfp, "class Model(nn.Module):\n");
    fprintf(pyfp, "    def __init__(self):\n");
    fprintf(pyfp, "        super(Model, self).__init__()\n");

    fprintf(pyfp, "\n");

    // module
    {
        for (const Operator* op : ops)
        {
            if (op->type.substr(0, 3) != "nn." && op->type.substr(0, 16) != "torchvision.ops.")
                continue;

            fprintf(pyfp, "        self.%s = %s(", sanitize_identifier(op->name).c_str(), op->type.c_str());

            int param_count = op->params.size();
            if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
            {
                param_count -= 2; // ignore scale and zero_point
            }

            int param_index = 0;
            for (const auto& it : op->params)
            {
                if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
                {
                    if (it.first == "scale" || it.first == "zero_point")
                        continue;
                }

                fprintf(pyfp, "%s=", it.first.c_str());

                const Parameter& param = it.second;
                if (param.type == 0)
                {
                    fprintf(pyfp, "None");
                }
                if (param.type == 1)
                {
                    if (param.b)
                        fprintf(pyfp, "True");
                    else
                        fprintf(pyfp, "False");
                }
                if (param.type == 2)
                {
                    fprintf(pyfp, "%d", param.i);
                }
                if (param.type == 3)
                {
                    fprintf(pyfp, "%f", param.f);
                }
                if (param.type == 4)
                {
                    if (param.s.substr(0, 6) == "torch.")
                    {
                        fprintf(pyfp, "%s", param.s.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "\'%s\'", param.s.c_str());
                    }
                }
                if (param.type == 5)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.ai.size(); i++)
                    {
                        if ((op->type == "nn.AdaptiveAvgPool2d"
                                || op->type == "nn.AdaptiveAvgPool3d"
                                || op->type == "nn.AdaptiveMaxPool2d"
                                || op->type == "nn.AdaptiveMaxPool3d")
                                && it.first == "output_size" && param.ai[i] == 0)
                        {
                            fprintf(pyfp, "None");
                        }
                        else
                        {
                            fprintf(pyfp, "%d", param.ai[i]);
                        }
                        if (i + 1 != param.ai.size() || param.ai.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }
                if (param.type == 6)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.af.size(); i++)
                    {
                        fprintf(pyfp, "%f", param.af[i]);
                        if (i + 1 != param.af.size() || param.af.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }
                if (param.type == 7)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.as.size(); i++)
                    {
                        if (param.as[i].substr(0, 6) == "torch.")
                        {
                            fprintf(pyfp, "%s", param.as[i].c_str());
                        }
                        else
                        {
                            fprintf(pyfp, "\'%s\'", param.as[i].c_str());
                        }
                        if (i + 1 != param.as.size() || param.as.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }

                param_index++;
                if (param_index != param_count)
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")\n");
        }
    }

    fprintf(pyfp, "\n");

    // load weights
    {
        fprintf(pyfp, "        archive = zipfile.ZipFile('%s', 'r')\n", pnnxbinpath.c_str());

        for (const Operator* op : ops)
        {
            if (op->type.substr(0, 3) != "nn." && op->type.substr(0, 16) != "torchvision.ops.")
                continue;

            if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
            {
                for (const auto& it : op->attrs)
                {
                    if (it.first == "weight" || it.first == "bias")
                    {
                        fprintf(pyfp, "        self_%s_%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                    }
                    else
                    {
                        // unknown attr
                        continue;
                    }

                    const Attribute& attr = it.second;
                    for (size_t i = 0; i < attr.shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", attr.shape[i]);
                        if (i + 1 != attr.shape.size())
                            fprintf(pyfp, ",");
                    }

                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }

                fprintf(pyfp, "        self.%s.set_weight_bias(self_%s_weight, self_%s_bias)\n", sanitize_identifier(op->name).c_str(), sanitize_identifier(op->name).c_str(), sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "        self.%s.scale = %f\n", sanitize_identifier(op->name).c_str(), op->params.at("scale").f);
                fprintf(pyfp, "        self.%s.zero_point = %d\n", sanitize_identifier(op->name).c_str(), op->params.at("zero_point").i);

                continue;
            }

            for (const auto& it : op->attrs)
            {
                if (it.first == "running_mean" || it.first == "running_var")
                {
                    fprintf(pyfp, "        self.%s.%s = self.load_pnnx_bin_as_tensor(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                }
                else
                {
                    fprintf(pyfp, "        self.%s.%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                }

                const Attribute& attr = it.second;
                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d", attr.shape[i]);
                    if (i + 1 != attr.shape.size())
                        fprintf(pyfp, ",");
                }

                if (attr.type == 1 || attr.type == 2 || attr.type == 3)
                {
                    fprintf(pyfp, "), '%s')\n", type_to_numpy_string(attr.type));
                }
                else
                {
                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }
            }
        }

        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Attribute")
                continue;

            const std::string& key = op->attrs.begin()->first;
            const Attribute& attr = op->attrs.begin()->second;

            bool is_running_mean_var = false;
            {
                const Operand* r = op->outputs[0];
                if (r->consumers.size() == 1)
                {
                    const Operator* op2 = r->consumers[0];
                    if (op2->type == "F.batch_norm" || op2->type == "F.instance_norm")
                    {
                        if (r == op2->inputs[1] || r == op2->inputs[2])
                        {
                            is_running_mean_var = true;
                        }
                    }
                }
            }

            bool is_empty = false;
            for (size_t i = 0; i < attr.shape.size(); i++)
            {
                if (attr.shape[i] == 0)
                    is_empty = true;
            }

            if (is_empty)
            {
                fprintf(pyfp, "        self.%s_%s = torch.from_numpy(np.empty((", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str());

                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d,", attr.shape[i]);
                }

                fprintf(pyfp, "), dtype='%s'))\n", type_to_numpy_string(attr.type));
            }
            else
            {
                if (is_running_mean_var)
                {
                    fprintf(pyfp, "        self.%s_%s = self.load_pnnx_bin_as_tensor(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str(), op->name.c_str(), key.c_str());
                }
                else
                {
                    fprintf(pyfp, "        self.%s_%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str(), op->name.c_str(), key.c_str());
                }

                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d,", attr.shape[i]);
                }

                if (attr.type == 1 || attr.type == 2 || attr.type == 3)
                {
                    fprintf(pyfp, "), '%s')\n", type_to_numpy_string(attr.type));
                }
                else
                {
                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }
            }
        }

        fprintf(pyfp, "        archive.close()\n");
    }

    fprintf(pyfp, "\n");

    // utility function
    {
        fprintf(pyfp, "    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):\n");
        fprintf(pyfp, "        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):\n");
        fprintf(pyfp, "        fd, tmppath = tempfile.mkstemp()\n");
        fprintf(pyfp, "        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:\n");
        fprintf(pyfp, "            tmpf.write(keyfile.read())\n");
        fprintf(pyfp, "        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()\n");
        fprintf(pyfp, "        os.remove(tmppath)\n");
        fprintf(pyfp, "        return torch.from_numpy(m)\n");
    }

    fprintf(pyfp, "\n");

    // def forward
    {
        fprintf(pyfp, "    def forward(self");

        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            fprintf(pyfp, ", v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
        }

        fprintf(pyfp, "):\n");
    }

    // forward body
    {
        for (const Operator* op : ops)
        {
            if (op->type == "pnnx.Input" || op->type == "pnnx.Output")
                continue;

            if (op->type == "pnnx.SliceIndexes")
                continue;

            fprintf(pyfp, "        ");

            if (op->type == "pnnx.Expression")
            {
                // expr
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                std::string expanded_expr = expand_expression(op);
                fprintf(pyfp, " = %s\n", expanded_expr.c_str());
            }
            else if (op->type == "pnnx.Attribute")
            {
                const std::string& key = op->attrs.begin()->first;
                fprintf(pyfp, "v_%s = self.%s_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str());
            }
            else if (op->type == "Tensor.slice")
            {
                // slice expr
                std::string slice_expr = make_slice_expression(op);
                fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), slice_expr.c_str());
            }
            else if (op->type == "Tensor.slice_copy")
            {
                // slice copy expr
                std::string slice_expr = make_slice_expression(op);
                fprintf(pyfp, "v_%s = v_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str());
                fprintf(pyfp, "        v_%s[%s] = v_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), slice_expr.c_str(), sanitize_identifier(op->inputs[1]->name).c_str());
            }
            else if (op->type == "Tensor.index")
            {
                // index expr
                // if (op->inputs.size() == 2)
                // {
                //     std::string expanded_expr = expand_expression(op->inputs[1]->producer);
                //     fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), expanded_expr.c_str());
                // }
                // else
                // {
                //     std::string index_expr = make_index_expression(op);
                //     fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), index_expr.c_str());
                // }
                std::string index_expr = make_index_expression(op);
                fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), index_expr.c_str());
                
            }
            else if (op->type == "Tensor.expand")
            {
                // expand
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& shape = op->params.at("shape").ai;
                    for (size_t i = 0; i < shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", shape[i]);
                        if (i + 1 != shape.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "Tensor.view" || op->type == "Tensor.reshape")
            {
                // view reshape
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& shape = op->params.at("shape").ai;
                    for (size_t i = 0; i < shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", shape[i]);
                        if (i + 1 != shape.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "Tensor.repeat")
            {
                // view reshape
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& sizes = op->params.at("sizes").ai;
                    for (size_t i = 0; i < sizes.size(); i++)
                    {
                        fprintf(pyfp, "%d", sizes[i]);
                        if (i + 1 != sizes.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.cat" || op->type == "torch.stack")
            {
                // cat
                fprintf(pyfp, "v_%s = %s(", sanitize_identifier(op->outputs[0]->name).c_str(), op->type.c_str());
                if (op->inputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < op->inputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                        if (i + 1 != op->inputs.size())
                            fprintf(pyfp, ", ");
                    }
                    fprintf(pyfp, ")");
                }
                fprintf(pyfp, ", dim=%d", op->params.at("dim").i);
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.einsum")
            {
                // einsum
                fprintf(pyfp, "v_%s = %s(", sanitize_identifier(op->outputs[0]->name).c_str(), op->type.c_str());

                fprintf(pyfp, "\'%s\'", op->params.at("equation").s.c_str());

                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, ", v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "prim::TupleUnpack")
            {
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = v_%s\n", sanitize_identifier(op->inputs[0]->name).c_str());
            }
            else if (op->type == "prim::TupleConstruct")
            {
                fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
                fprintf(pyfp, " = (");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "prim::ListUnpack")
            {
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = v_%s\n", sanitize_identifier(op->inputs[0]->name).c_str());
            }
            else if (op->type == "prim::ListConstruct")
            {
                fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
                fprintf(pyfp, " = [");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                    if (i + 1 != op->inputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "]\n");
            }
            else if (op->type == "nn.GRU" || op->type == "nn.RNN")
            {
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "v_%s, v_%s", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->outputs[1]->name).c_str());
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, ", v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "nn.LSTM")
            {
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "v_%s, (v_%s, v_%s)", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->outputs[1]->name).c_str(), sanitize_identifier(op->outputs[2]->name).c_str());
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                if (op->inputs.size() == 3)
                {
                    fprintf(pyfp, ", (v_%s, v_%s)", sanitize_identifier(op->inputs[1]->name).c_str(), sanitize_identifier(op->inputs[2]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "nn.MultiheadAttention")
            {
                bool need_weights = true;
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                    need_weights = false;
                }
                else
                {
                    for (size_t i = 0; i < op->outputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                        if (i + 1 != op->outputs.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                if (op->inputs.size() == 1)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in0.c_str(), in0.c_str());
                }
                else if (op->inputs.size() == 2)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    if (op->inputnames.size() == 2 && op->inputnames[1] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in0.c_str(), in0.c_str(), in1.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in1.c_str());
                    }
                }
                else if (op->inputs.size() == 3)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    std::string in2 = sanitize_identifier(op->inputs[2]->name);
                    if (op->inputnames.size() == 3 && op->inputnames[2] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in1.c_str(), in1.c_str(), in2.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in2.c_str());
                    }
                }
                else if (op->inputs.size() == 4)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    std::string in2 = sanitize_identifier(op->inputs[2]->name);
                    std::string in3 = sanitize_identifier(op->inputs[3]->name);
                    if (op->inputnames.size() == 4 && op->inputnames[3] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in1.c_str(), in2.c_str(), in3.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in2.c_str(), in3.c_str());
                    }
                }
                else
                {
                    for (size_t i = 0; i < op->inputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                        if (i + 1 != op->inputs.size())
                            fprintf(pyfp, ", ");
                    }
                }
                if (need_weights)
                {
                    fprintf(pyfp, ", need_weights=True");
                }
                else
                {
                    fprintf(pyfp, ", need_weights=False");
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type.substr(0, 3) == "nn." || op->type.substr(0, 16) == "torchvision.ops.")
            {
                // self.xxx()
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                    if (i + 1 != op->inputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type.find("::") != std::string::npos || op->type.find(".") != std::string::npos)
            {
                // direct
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }

                if (op->type.substr(0, 7) == "Tensor.")
                {
                    if (op->type == "Tensor.fill")
                    {
                        fprintf(pyfp, " = v_%s.fill_(", sanitize_identifier(op->inputs[0]->name).c_str());
                    }
                    else
                    {
                        fprintf(pyfp, " = v_%s.%s(", sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                    }

                    if (op->inputnames.size() == op->inputs.size())
                    {
                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            if (!op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                        }

                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            if (op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "%s=v_%s, ", op->inputnames[i].c_str(), sanitize_identifier(op->inputs[i]->name).c_str());
                        }
                    }
                    else
                    {
                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                        }
                    }
                }
                else
                {
                    fprintf(pyfp, " = %s(", op->type.c_str());

                    if (op->inputnames.size() == op->inputs.size())
                    {
                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            if (!op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }

                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            if (op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "%s=v_%s", op->inputnames[i].c_str(), sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }
                    }
                }

                int i = 0;
                for (const auto& it : op->params)
                {
                    if (op->type.substr(0, 7) == "Tensor." && i == 0)
                    {
                        fprintf(pyfp, "%s=", it.first.c_str());
                    }
                    else if (op->inputs.empty() && i == 0)
                    {
                        fprintf(pyfp, "%s=", it.first.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, ", %s=", it.first.c_str());
                    }

                    i++;

                    const Parameter& param = it.second;
                    if (param.type == 0)
                    {
                        if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            fprintf(pyfp, "torch.tensor(False)");
                        }
                        else
                        {
                            fprintf(pyfp, "None");
                        }
                    }
                    if (param.type == 1)
                    {
                        if (param.b)
                            fprintf(pyfp, "True");
                        else
                            fprintf(pyfp, "False");
                    }
                    if (param.type == 2)
                    {
                        if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            fprintf(pyfp, "torch.tensor(%d)", param.i);
                        }
                        else
                        {
                            fprintf(pyfp, "%d", param.i);
                        }
                    }
                    if (param.type == 3)
                    {
                        if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            fprintf(pyfp, "torch.tensor(%f)", param.f);
                        }
                        else
                        {
                            fprintf(pyfp, "%f", param.f);
                        }
                    }
                    if (param.type == 4)
                    {
                        if (param.s.substr(0, 6) == "torch.")
                        {
                            fprintf(pyfp, "%s", param.s.c_str());
                        }
                        else if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            if (param.s == "inf" || param.s == "-inf")
                            {
                                fprintf(pyfp, "torch.tensor(float(\'%s\'))", param.s.c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "torch.tensor(\'%s\')", param.s.c_str());
                            }
                        }
                        else
                        {
                            if (param.s == "inf" || param.s == "-inf")
                            {
                                fprintf(pyfp, "float(\'%s\')", param.s.c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "\'%s\'", param.s.c_str());
                            }
                        }
                    }
                    if (param.type == 5)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.ai.size(); i++)
                        {
                            if ((op->type == "F.adaptive_avg_pool2d"
                                    || op->type == "F.adaptive_avg_pool3d"
                                    || op->type == "F.adaptive_max_pool2d"
                                    || op->type == "F.adaptive_max_pool3d")
                                    && it.first == "output_size" && param.ai[i] == 0)
                            {
                                fprintf(pyfp, "None");
                            }
                            else
                            {
                                fprintf(pyfp, "%d", param.ai[i]);
                            }
                            if (i + 1 != param.ai.size() || param.ai.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 6)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.af.size(); i++)
                        {
                            fprintf(pyfp, "%f", param.af[i]);
                            if (i + 1 != param.af.size() || param.af.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 7)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.as.size(); i++)
                        {
                            if (param.as[i].substr(0, 6) == "torch.")
                            {
                                fprintf(pyfp, "%s", param.as[i].c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "\'%s\'", param.as[i].c_str());
                            }
                            if (i + 1 != param.as.size() || param.as.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 10)
                    {
                        fprintf(pyfp, "(%f%+fj)", param.c.real(), param.c.imag());
                    }
                    if (param.type == 11)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.ac.size(); i++)
                        {
                            fprintf(pyfp, "(%f%+fj)", param.ac[i].real(), param.ac[i].imag());
                            if (i + 1 != param.ac.size() || param.ac.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                }

                fprintf(pyfp, ")\n");
            }
            else
            {
                fprintf(stderr, "todo %s\n", op->type.c_str());
            }
        }
    }

    // return
    {
        fprintf(pyfp, "        return ");

        int output_count = 0;
        {
            for (const Operator* op : ops)
            {
                if (op->type == "pnnx.Output")
                    output_count++;
            }
        }

        int output_index = 0;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Output")
                continue;

            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
            if (output_index + 1 != output_count)
                fprintf(pyfp, ", ");

            output_index++;
        }

        fprintf(pyfp, "\n");
    }

    fprintf(pyfp, "\n");

    // export torchscript
    {
        fprintf(pyfp, "def export_torchscript():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    mod = torch.jit.trace(net, %s)\n", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    mod = torch.jit.trace(net, (");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, "))\n");
        }

        fprintf(pyfp, "    mod.save(\"%s.pt\")\n", pypath.c_str());
    }

    fprintf(pyfp, "\n");

    // export onnx
    {
        fprintf(pyfp, "def export_onnx():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        // torch.onnx._export(net, v_0, "test_swin_t.onnx", export_params=True, opset_version=14, input_names=['in0'], output_names=['out0'])

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    torch.onnx._export(net, %s", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    torch.onnx._export(net, (");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")");
        }

        fprintf(pyfp, ", \"%s.onnx\", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13", pypath.c_str());

        fprintf(pyfp, ", input_names=[");
        {
            int input_count = 0;
            {
                for (const Operator* op : ops)
                {
                    if (op->type == "pnnx.Input")
                        input_count++;
                }
            }

            int input_index = 0;
            for (const Operator* op : ops)
            {
                if (op->type != "pnnx.Input")
                    continue;

                fprintf(pyfp, "'in%d'", input_index);
                if (input_index + 1 != input_count)
                    fprintf(pyfp, ", ");

                input_index++;
            }
        }
        fprintf(pyfp, "]");

        fprintf(pyfp, ", output_names=[");
        {
            int output_count = 0;
            {
                for (const Operator* op : ops)
                {
                    if (op->type == "pnnx.Output")
                        output_count++;
                }
            }

            int output_index = 0;
            for (const Operator* op : ops)
            {
                if (op->type != "pnnx.Output")
                    continue;

                fprintf(pyfp, "'out%d'", output_index);
                if (output_index + 1 != output_count)
                    fprintf(pyfp, ", ");

                output_index++;
            }
        }
        fprintf(pyfp, "]");

        fprintf(pyfp, ")\n");
    }

    fprintf(pyfp, "\n");

    // test inference
    {
        fprintf(pyfp, "def test_inference():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    return net(%s)\n", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    return net(");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")\n");
        }
    }

    fprintf(pyfp, "\n");

    // main
    {
        fprintf(pyfp, "if __name__ == \"__main__\":\n");
        fprintf(pyfp, "    print(test_inference())\n");
    }

    fclose(pyfp);

    return 0;
}

// add by senli split string
std::vector<std::string> split_string(const std::string& s, const std::string& sub_s)
{
    std::vector<std::string> tokens;
    std::string token;
    char delimiter = sub_s[0];
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// add by senli match function and insert
int insert_function(FILE* pyfp, std::vector<std::string>& custom_ops_names, std::string& customop_infer_py)
{
    std::ifstream file(customop_infer_py);

    if (!file)
    {
        std::cout << "can not open customop_infer_py" << std::endl;
        return -1;
    }

    std::string line;
    std::ostringstream functionContent;
    bool insideFunction = false;

    std::regex functionStartRegex(R"(\s*def\s+(\w+)\s*\(.*\):)");
    std::regex pattern("^import");
    std::smatch match;

    while (std::getline(file, line))
    {
        if (std::regex_search(line, match, pattern))
        {
            functionContent << line << std::endl;
        }
        if (std::regex_search(line, match, functionStartRegex))
        {
            bool found = std::find(custom_ops_names.begin(), custom_ops_names.end(), match[1]) != custom_ops_names.end();
            if (found)
            {
                insideFunction = true;
                functionContent << line << std::endl;
            }
            else
            {
                insideFunction = false; // if function not in custom_ops_names, stop get current content
            }
        }
        else if (insideFunction)
        {
            // if not meet th end of func, continue get content
            functionContent << line << std::endl;
        }
    }

    file.close();

    if (insideFunction)
    {
        std::cout << "the content of func:" << std::endl
                  << functionContent.str() << std::endl;
    }
    else
    {
        std::cout << "can not find func" << std::endl;
    }

    std::string str = functionContent.str();
    std::istringstream iss(str);
    std::string line1;
    std::vector<std::string> lines;

    while (std::getline(iss, line1))
    {
        fprintf(pyfp, "%s", (line1 + "\n").c_str());
    }

    return 1;
}

// add by senli get all custom_op_names 20240320
int get_custom_op_names(std::string& customop_infer_py, std::vector<std::string>& custom_ops_names)
{
    std::ifstream file(customop_infer_py);

    if (!file)
    {
        std::cout << "can not open customop_infer_py" << std::endl;
        return -1;
    }

    std::string line;

    std::regex functionStartRegex(R"(\s*def\s+(\w+)\s*\(.*\):)");
    std::smatch match;

    while (std::getline(file, line))
    {
        if (std::regex_search(line, match, functionStartRegex))
        {
            custom_ops_names.push_back(match[1]);
        }
    }

    file.close();
    return 1;
}

// add by senli 20240320 get Directory and file_name of customop_infer_py
std::vector<std::string> getDirectoryPath(const std::string& filePath)
{
    std::vector<std::string> customop_infer_py_info;
    size_t found = filePath.find_last_of("/\\");
    if (found != std::string::npos)
    {
        std::string Directory = filePath.substr(0, found);
        customop_infer_py_info.push_back(Directory);
        std::string file_name = filePath.substr(found + 1);
        size_t dotPosition = file_name.rfind('.'); // 从后向前查找最后一个点
        if (dotPosition != std::string::npos)
        {
            customop_infer_py_info.push_back(file_name.substr(0, dotPosition));
        }
    }
    return customop_infer_py_info;
}

int Graph::python_infer(const std::string& pypath, const std::string& binpath,
                        const std::vector<std::string>& customop_modules, std::set<std::string>& custom_ops,
                        std::string& customop_infer_py,
                        std::string& save_dir)
{
    FILE* pyfp = fopen(pypath.c_str(), "wb");
    if (!pyfp)
    {
        fprintf(stderr, "fopen %s failed\n", pypath.c_str());
        return -1;
    }

    fprintf(pyfp, "import os\n");
    fprintf(pyfp, "import sys\n");
    fprintf(pyfp, "import numpy as np\n");
    fprintf(pyfp, "import tempfile, zipfile\n");
    fprintf(pyfp, "import torch\n");
    fprintf(pyfp, "import torch.nn as nn\n");
    fprintf(pyfp, "import torch.nn.functional as F\n");
    fprintf(pyfp, "import importlib\n");
    fprintf(pyfp, "try:\n");
    fprintf(pyfp, "    import torchvision\n");
    fprintf(pyfp, "except:\n");
    fprintf(pyfp, "    pass\n");

    fprintf(pyfp, "\n");
    //add by senli import customop_infer_py 20240320
    std::vector<std::string> custom_ops_names;
    //add by senli 2024015 load custom_op lib
    if (customop_infer_py == "None")
    {
        for (auto m : customop_modules)
        {
#ifdef _WIN32
            fprintf(pyfp, "torch.ops.load_library(r'%s", m.c_str());
#elif defined(__linux__)
            fprintf(pyfp, "torch.ops.load_library('%s", m.c_str());
#elif defined(__APPLE__)
            fprintf(pyfp, "torch.ops.load_library('%s", m.c_str());
#endif

            fprintf(pyfp, "')\n");
        }
    }
    else
    {
        // add by senli insert custom_op_infer
        // std::vector<std::string> custom_ops_names;
        // for (const auto& custom_op : custom_ops) {
        //     std::vector<std::string> tokens = split_string(custom_op, ".");
        //     std::reverse(tokens.begin(), tokens.end());
        //     std::string custom_op_name = tokens.at(0);
        //     custom_ops_names.push_back(custom_op_name);
        // }
        // int insert_flag = insert_function(pyfp, custom_ops_names, customop_infer_py);
        // if(insert_flag == -1)
        // {
        //     std::cerr << "please check th path of customop_infer_py" << std::endl;
        //     return -1;
        // }

        //add by senli import customop_infer_py 20240320
        int insert_flag = get_custom_op_names(customop_infer_py, custom_ops_names);
        if (insert_flag == -1)
        {
            std::cerr << "failed in  get_custom_op_names" << std::endl;
            return -1;
        }
        std::vector<std::string> customop_infer_py_info = getDirectoryPath(customop_infer_py);
        if (customop_infer_py_info.size() == 0 || customop_infer_py_info.size() == 1)
        {
            std::cerr << "please check th path of customop_infer_py" << std::endl;
            return -1;
        }
#ifdef _WIN32
        fprintf(pyfp, "sys.path.append(r'%s", customop_infer_py_info[0].c_str());
#elif defined(__linux__)
        fprintf(pyfp, "sys.path.append('%s", customop_infer_py_info[0].c_str());
#elif defined(__APPLE__)
        fprintf(pyfp, "sys.path.append('%s", customop_infer_py_info[0].c_str());
#endif
        fprintf(pyfp, "')\n");
        fprintf(pyfp, "from %s import *\n", customop_infer_py_info[1].c_str());
    }
    fprintf(pyfp, "\n");

    // load_module
    {
        fprintf(pyfp, "def load_module(module_path):\n");
        fprintf(pyfp, "    spec = importlib.util.spec_from_file_location('module', module_path)\n");
        fprintf(pyfp, "    module = importlib.util.module_from_spec(spec)\n");
        fprintf(pyfp, "    spec.loader.exec_module(module)\n");
        fprintf(pyfp, "    return module\n");
        fprintf(pyfp, "\n");
    }
    //add by senli[pnnx_infer]
    fprintf(pyfp, "class Model(nn.Module):\n");
    fprintf(pyfp, "    def __init__(self, bin_path, infer_flag = False):\n");
    fprintf(pyfp, "        super(Model, self).__init__()\n");

    fprintf(pyfp, "\n");

    // module
    {
        //add by senli[pnnx_infer]
        fprintf(pyfp, "        self.infer_flag = infer_flag\n");
        for (const Operator* op : ops)
        {
            if(op->type == "pnnx.Loop" || op->type == "pnnx.If")
            {
                std::string op_name = op->name;
                std::vector<std::string> block_names = op->params.at("block_names").as;
                for(auto op_name: block_names)
                {
                    std::string subModelBinPath = save_dir + "/" + op_name + ".pnnx.bin";
                    std::string subModelInferPath = save_dir + "/" + op_name + "_pnnx_infer.py";
                    fprintf(pyfp, "        %s = load_module('%s')\n", (op_name + "_Mod").c_str(), subModelInferPath.c_str());
                    fprintf(pyfp, "        %s = getattr(%s, 'Model')\n", (op_name + "_Cls").c_str(), (op_name + "_Mod").c_str());
                    fprintf(pyfp, "        %s = %s('%s', True)\n", ("self." + op_name + "_Obj").c_str(), (op_name + "_Cls").c_str(),  subModelBinPath.c_str());
                    fprintf(pyfp, "        %s.eval()\n", ("self." + op_name + "_Obj").c_str());
                }
                continue;
            }
            if (op->type.substr(0, 3) != "nn." && op->type.substr(0, 16) != "torchvision.ops.")
                continue;

            fprintf(pyfp, "        self.%s = %s(", sanitize_identifier(op->name).c_str(), op->type.c_str());

            int param_count = op->params.size();
            if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
            {
                param_count -= 2; // ignore scale and zero_point
            }

            int param_index = 0;
            for (const auto& it : op->params)
            {
                if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
                {
                    if (it.first == "scale" || it.first == "zero_point")
                        continue;
                }

                fprintf(pyfp, "%s=", it.first.c_str());

                const Parameter& param = it.second;
                if (param.type == 0)
                {
                    fprintf(pyfp, "None");
                }
                if (param.type == 1)
                {
                    if (param.b)
                        fprintf(pyfp, "True");
                    else
                        fprintf(pyfp, "False");
                }
                if (param.type == 2)
                {
                    fprintf(pyfp, "%d", param.i);
                }
                if (param.type == 3)
                {
                    fprintf(pyfp, "%f", param.f);
                }
                if (param.type == 4)
                {
                    if (param.s.substr(0, 6) == "torch.")
                    {
                        fprintf(pyfp, "%s", param.s.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "\'%s\'", param.s.c_str());
                    }
                }
                if (param.type == 5)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.ai.size(); i++)
                    {
                        if ((op->type == "nn.AdaptiveAvgPool2d"
                                || op->type == "nn.AdaptiveAvgPool3d"
                                || op->type == "nn.AdaptiveMaxPool2d"
                                || op->type == "nn.AdaptiveMaxPool3d")
                                && it.first == "output_size" && param.ai[i] == 0)
                        {
                            fprintf(pyfp, "None");
                        }
                        else
                        {
                            fprintf(pyfp, "%d", param.ai[i]);
                        }
                        if (i + 1 != param.ai.size() || param.ai.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }
                if (param.type == 6)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.af.size(); i++)
                    {
                        fprintf(pyfp, "%f", param.af[i]);
                        if (i + 1 != param.af.size() || param.af.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }
                if (param.type == 7)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.as.size(); i++)
                    {
                        if (param.as[i].substr(0, 6) == "torch.")
                        {
                            fprintf(pyfp, "%s", param.as[i].c_str());
                        }
                        else
                        {
                            fprintf(pyfp, "\'%s\'", param.as[i].c_str());
                        }
                        if (i + 1 != param.as.size() || param.as.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }

                param_index++;
                if (param_index != param_count)
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")\n");
        }
    }

    fprintf(pyfp, "\n");

    // load weights
    //add by senli
    {
        fprintf(pyfp, "        archive = zipfile.ZipFile(bin_path, 'r')\n"); //fix to bin_path

        for (const Operator* op : ops)
        {
            if (op->type.substr(0, 3) != "nn." && op->type.substr(0, 16) != "torchvision.ops.")
                continue;

            if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
            {
                for (const auto& it : op->attrs)
                {
                    if (it.first == "weight" || it.first == "bias")
                    {
                        fprintf(pyfp, "        self_%s_%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                    }
                    else
                    {
                        // unknown attr
                        continue;
                    }

                    const Attribute& attr = it.second;
                    for (size_t i = 0; i < attr.shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", attr.shape[i]);
                        if (i + 1 != attr.shape.size())
                            fprintf(pyfp, ",");
                    }

                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }

                fprintf(pyfp, "        self.%s.set_weight_bias(self_%s_weight, self_%s_bias)\n", sanitize_identifier(op->name).c_str(), sanitize_identifier(op->name).c_str(), sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "        self.%s.scale = %f\n", sanitize_identifier(op->name).c_str(), op->params.at("scale").f);
                fprintf(pyfp, "        self.%s.zero_point = %d\n", sanitize_identifier(op->name).c_str(), op->params.at("zero_point").i);

                continue;
            }

            for (const auto& it : op->attrs)
            {
                if (it.first == "running_mean" || it.first == "running_var")
                {
                    fprintf(pyfp, "        self.%s.%s = self.load_pnnx_bin_as_tensor(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                }
                else
                {
                    fprintf(pyfp, "        self.%s.%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                }

                const Attribute& attr = it.second;
                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d", attr.shape[i]);
                    if (i + 1 != attr.shape.size())
                        fprintf(pyfp, ",");
                }

                if (attr.type == 1 || attr.type == 2 || attr.type == 3)
                {
                    fprintf(pyfp, "), '%s')\n", type_to_numpy_string(attr.type));
                }
                else
                {
                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }
            }
        }

        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Attribute")
                continue;

            const std::string& key = op->attrs.begin()->first;
            const Attribute& attr = op->attrs.begin()->second;

            bool is_running_mean_var = false;
            {
                const Operand* r = op->outputs[0];
                if (r->consumers.size() == 1)
                {
                    const Operator* op2 = r->consumers[0];
                    if (op2->type == "F.batch_norm" || op2->type == "F.instance_norm")
                    {
                        if (r == op2->inputs[1] || r == op2->inputs[2])
                        {
                            is_running_mean_var = true;
                        }
                    }
                }
            }

            bool is_empty = false;
            for (size_t i = 0; i < attr.shape.size(); i++)
            {
                if (attr.shape[i] == 0)
                    is_empty = true;
            }

            if (is_empty)
            {
                fprintf(pyfp, "        self.%s_%s = torch.from_numpy(np.empty((", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str());

                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d,", attr.shape[i]);
                }

                fprintf(pyfp, "), dtype='%s'))\n", type_to_numpy_string(attr.type));
            }
            else
            {
                if (is_running_mean_var)
                {
                    fprintf(pyfp, "        self.%s_%s = self.load_pnnx_bin_as_tensor(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str(), op->name.c_str(), key.c_str());
                }
                else
                {
                    fprintf(pyfp, "        self.%s_%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str(), op->name.c_str(), key.c_str());
                }

                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d,", attr.shape[i]);
                }

                if (attr.type == 1 || attr.type == 2 || attr.type == 3)
                {
                    fprintf(pyfp, "), '%s')\n", type_to_numpy_string(attr.type));
                }
                else
                {
                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }
            }
        }

        fprintf(pyfp, "        archive.close()\n");
    }

    fprintf(pyfp, "\n");

    // get input_shape and input_type add by senli[pnnx_infer]
    {
        // get shape and type of the input op 
        std::vector<std::vector<int>> input_shapes;
        std::vector<std::string> input_types;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;
            const Operand* r = op->outputs[0];
            input_shapes.push_back(r->shape);
            input_types.push_back(type_to_string(r->type));
        }

        //insert shape
        // example:
        // def getInput(self,):
        //     return [[1, 3, 32, 32],[1,3,64,64]]

        fprintf(pyfp, "    def getInput(self,):\n");
        fprintf(pyfp, "        return [");
        
        for (size_t i = 0; i < input_shapes.size(); i++)
        {
            std::vector<int> one_input_shape = input_shapes[i];
            fprintf(pyfp, "[");
            for (size_t j = 0; j < one_input_shape.size(); j++)
            {
                fprintf(pyfp, "%d", one_input_shape[j]);
                if (j + 1 != one_input_shape.size())
                    fprintf(pyfp, ", ");
            }
            fprintf(pyfp, "]");
            if (i + 1 != input_shapes.size())
                fprintf(pyfp, ", ");
        }
        fprintf(pyfp, "]\n");

        fprintf(pyfp, "\n");
        
        //insert type
        // example:
        // def getInputType(self,):
        //     return ['fp32','i64']
        fprintf(pyfp, "    def getInputType(self,):\n");
        fprintf(pyfp, "        return [");
        for (size_t i = 0; i < input_types.size(); i++)
        {
            std::string input_type = input_types[i];
            fprintf(pyfp, "'%s'", input_type.c_str());
            if (i + 1 != input_types.size())
                fprintf(pyfp, ", ");
        }
        fprintf(pyfp, "]\n");

    }
    fprintf(pyfp, "\n");
    // utility function
    {
        fprintf(pyfp, "    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):\n");
        fprintf(pyfp, "        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):\n");
        fprintf(pyfp, "        fd, tmppath = tempfile.mkstemp()\n");
        fprintf(pyfp, "        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:\n");
        fprintf(pyfp, "            tmpf.write(keyfile.read())\n");
        fprintf(pyfp, "        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()\n");
        fprintf(pyfp, "        os.remove(tmppath)\n");
        fprintf(pyfp, "        return torch.from_numpy(m)\n");
    }

    fprintf(pyfp, "\n");

    // def forward
    {
        fprintf(pyfp, "    def forward(self");

        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            fprintf(pyfp, ", v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
        }

        fprintf(pyfp, "):\n");
    }

    // forward body
    {
        for (const Operator* op : ops)
        {
            
            if (op->type == "pnnx.Input" || op->type == "pnnx.Output")
                continue;

            if (op->type == "pnnx.SliceIndexes")
                continue;

            fprintf(pyfp, "        ");

            if(op->type == "pnnx.Loop")
            {
                std::string condition_expr = op->params.at("condition").s;
                int iter_num = op->params.at("iter_num").i;
                // std::vector<std::string> block_names = op->params.at("block_names").as;
                std::string op_name = op->name;
                std::vector<Operand*> inputs = op->inputs;
                std::vector<Operand*> outputs = op->outputs;
                std::string output_list = "";
                std::string input_list = "";
                std::string real_input_list = "";
                for(int index = 0; index < outputs.size(); index++)
                {
                    std::string cur_output_name = sanitize_identifier(op->outputs[index]->name);
                    std::string cur_input_name = sanitize_identifier(op->inputs[index]->name);
                    output_list  = output_list + "v_" + cur_output_name;
                    input_list  = input_list + "v_" + cur_input_name;
                    if (index + 1 != outputs.size())
                    {
                        output_list = output_list + ", ";
                        input_list = input_list + ", ";
                    }
                        
                }
                for(int index = 0; index < inputs.size(); index++)
                {
                    std::string cur_input_name = sanitize_identifier(op->inputs[index]->name);
                    real_input_list  = real_input_list + "v_" + cur_input_name;
                    if (index + 1 != inputs.size())
                        real_input_list = real_input_list + ", ";
                }
                fprintf(pyfp, "%s = %s\n", output_list.c_str(), input_list.c_str());
                fprintf(pyfp, "        condition = %s\n", condition_expr.c_str());
                fprintf(pyfp, "        i = 0\n");
                fprintf(pyfp, "        while condition and i < %s:\n", std::to_string(iter_num).c_str());
                fprintf(pyfp, "            %s = %s\n", input_list.c_str(), output_list.c_str());
                fprintf(pyfp, "            %s = %s(%s)\n", output_list.c_str(), ("self." + op_name + "_Obj").c_str(), real_input_list.c_str());
                fprintf(pyfp, "            i += 1\n");
                continue;
            }

            if(op->type == "pnnx.If")
            {
                std::string op_name = op->name;
                std::vector<Operand*> inputs = op->inputs;
                std::vector<Operand*> outputs = op->outputs;
                std::vector<std::string> block_names = op->params.at("block_names").as;
                std::unordered_map<std::string, std::string> block_input_indexes_map;
                for(auto block_name: block_names)
                {
                    std::string block_input_indexes_name = block_name + "_input_indexes";
                    std::vector<int> block_input_indexes = op->params.at(block_input_indexes_name).ai;
                    std::string input_list = "";
                    int index = 0;
                    int index_num = block_input_indexes.size();
                    for(auto input_index: block_input_indexes)
                    {
                        std::string cur_input_name = sanitize_identifier(op->inputs[input_index]->name);
                        input_list  = input_list + "v_" + cur_input_name;
                        if (index + 1 != index_num)
                        {
                            input_list = input_list + ", ";
                        }
                    }
                    block_input_indexes_map[block_name] = input_list;
                }

                std::string output_list = "";
                std::string real_input_list = "";
                for(int index = 0; index < outputs.size(); index++)
                {
                    std::string cur_output_name = sanitize_identifier(outputs[index]->name);
                    std::string cur_real_input_name = sanitize_identifier(inputs[index + 1]->name);
                    output_list  = output_list + "v_" + cur_output_name;
                    real_input_list  = real_input_list + "v_" + cur_real_input_name;
                    if (index + 1 != outputs.size())
                    {
                        output_list = output_list + ", ";
                        real_input_list = real_input_list + ", ";
                    }
                        
                }
                std::string condition = "v_" + sanitize_identifier(op->inputs[0]->name);
                fprintf(pyfp, "if(%s):\n", condition.c_str());
                fprintf(pyfp, "            %s = %s(%s)\n", output_list.c_str(), ("self." + block_names[0] + "_Obj").c_str(), block_input_indexes_map[block_names[0]].c_str());
                if(block_names.size() == 2)
                {
                    fprintf(pyfp, "        else:\n");
                    fprintf(pyfp, "            %s = %s(%s)\n", output_list.c_str(), ("self." + block_names[1] + "_Obj").c_str(), block_input_indexes_map[block_names[1]].c_str());
                }
                else
                {
                    fprintf(pyfp, "        else:\n");
                    fprintf(pyfp, "            %s = %s\n", output_list.c_str(), real_input_list.c_str());
                }
                continue;
            }


            if (op->type == "pnnx.Expression")
            {
                // expr
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                std::string expanded_expr = expand_expression(op);
                fprintf(pyfp, " = %s\n", expanded_expr.c_str());
            }
            else if (op->type == "pnnx.Attribute")
            {
                const std::string& key = op->attrs.begin()->first;
                fprintf(pyfp, "v_%s = self.%s_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str());
            }
            else if (op->type == "Tensor.slice")
            {
                // slice expr
                std::string slice_expr = make_slice_expression(op);
                fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), slice_expr.c_str());
            }
            else if (op->type == "Tensor.slice_copy")
            {
                // slice copy expr
                std::string slice_expr = make_slice_expression(op);
                fprintf(pyfp, "v_%s = v_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str());
                fprintf(pyfp, "        v_%s[%s] = v_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), slice_expr.c_str(), sanitize_identifier(op->inputs[1]->name).c_str());
            }
            else if (op->type == "Tensor.index")
            {
                // index expr
                // if (op->inputs.size() == 2)
                // {
                //     std::string expanded_expr = expand_expression(op->inputs[1]->producer);
                //     fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), expanded_expr.c_str());
                // }
                // else
                // {
                //     std::string index_expr = make_index_expression(op);
                //     fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), index_expr.c_str());
                // }
                std::string index_expr = make_index_expression(op);
                fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), index_expr.c_str());
            }
            else if (op->type == "Tensor.expand")
            {
                // expand
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& shape = op->params.at("shape").ai;
                    for (size_t i = 0; i < shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", shape[i]);
                        if (i + 1 != shape.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "Tensor.view" || op->type == "Tensor.reshape")
            {
                // view reshape
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& shape = op->params.at("shape").ai;
                    for (size_t i = 0; i < shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", shape[i]);
                        if (i + 1 != shape.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "Tensor.repeat")
            {
                // view reshape
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& sizes = op->params.at("sizes").ai;
                    for (size_t i = 0; i < sizes.size(); i++)
                    {
                        fprintf(pyfp, "%d", sizes[i]);
                        if (i + 1 != sizes.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.cat" || op->type == "torch.stack")
            {
                // cat
                fprintf(pyfp, "v_%s = %s(", sanitize_identifier(op->outputs[0]->name).c_str(), op->type.c_str());
                if (op->inputs.size() == 1)
                {
                    fprintf(pyfp, "[v_%s]", sanitize_identifier(op->inputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < op->inputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                        if (i + 1 != op->inputs.size())
                            fprintf(pyfp, ", ");
                    }
                    fprintf(pyfp, ")");
                }
                fprintf(pyfp, ", dim=%d", op->params.at("dim").i);
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.einsum")
            {
                // einsum
                fprintf(pyfp, "v_%s = %s(", sanitize_identifier(op->outputs[0]->name).c_str(), op->type.c_str());

                fprintf(pyfp, "\'%s\'", op->params.at("equation").s.c_str());

                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, ", v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "prim::TupleUnpack")
            {
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = v_%s\n", sanitize_identifier(op->inputs[0]->name).c_str());
            }
            else if (op->type == "prim::TupleConstruct")
            {
                fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
                fprintf(pyfp, " = (");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "prim::ListUnpack")
            {
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = v_%s\n", sanitize_identifier(op->inputs[0]->name).c_str());
            }
            else if (op->type == "prim::ListConstruct")
            {
                fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
                fprintf(pyfp, " = [");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                    if (i + 1 != op->inputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "]\n");
            }
            else if (op->type == "nn.GRU" || op->type == "nn.RNN")
            {
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "v_%s, v_%s", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->outputs[1]->name).c_str());
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, ", v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "nn.LSTM")
            {
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "v_%s, (v_%s, v_%s)", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->outputs[1]->name).c_str(), sanitize_identifier(op->outputs[2]->name).c_str());
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                if (op->inputs.size() == 3)
                {
                    fprintf(pyfp, ", (v_%s, v_%s)", sanitize_identifier(op->inputs[1]->name).c_str(), sanitize_identifier(op->inputs[2]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "nn.MultiheadAttention")
            {
                bool need_weights = true;
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                    need_weights = false;
                }
                else
                {
                    for (size_t i = 0; i < op->outputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                        if (i + 1 != op->outputs.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                if (op->inputs.size() == 1)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in0.c_str(), in0.c_str());
                }
                else if (op->inputs.size() == 2)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    if (op->inputnames.size() == 2 && op->inputnames[1] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in0.c_str(), in0.c_str(), in1.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in1.c_str());
                    }
                }
                else if (op->inputs.size() == 3)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    std::string in2 = sanitize_identifier(op->inputs[2]->name);
                    if (op->inputnames.size() == 3 && op->inputnames[2] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in1.c_str(), in1.c_str(), in2.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in2.c_str());
                    }
                }
                else if (op->inputs.size() == 4)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    std::string in2 = sanitize_identifier(op->inputs[2]->name);
                    std::string in3 = sanitize_identifier(op->inputs[3]->name);
                    if (op->inputnames.size() == 4 && op->inputnames[3] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in1.c_str(), in2.c_str(), in3.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in2.c_str(), in3.c_str());
                    }
                }
                else
                {
                    for (size_t i = 0; i < op->inputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                        if (i + 1 != op->inputs.size())
                            fprintf(pyfp, ", ");
                    }
                }
                if (need_weights)
                {
                    fprintf(pyfp, ", need_weights=True");
                }
                else
                {
                    fprintf(pyfp, ", need_weights=False");
                }
                fprintf(pyfp, ")\n");
            }
            //add by senli
            else if (op->type == "F.max_unpool2d")
            {
                /*
                根据输入和输出的shape，计算kernel_size,stride,padding
                以下为转换公式
                    padding = 0
                    stride = floor ( (output_size / input_size) )
                    kernel_size = output_size − (input_size - 1) * stride
                */
                std::vector<int> inshape = op->inputs[0]->shape;
                std::vector<int> outshape = op->outputs[0]->shape;
                int ih = inshape[2];
                int iw = inshape[3];
                int oh = outshape[2];
                int ow = outshape[3];
                int stride_h = std::floor(oh / ih);
                int stride_w = std::floor(ow / iw);
                int kernel_size_h = oh - (ih - 1) * stride_h;
                int kernel_size_w = ow - (iw - 1) * stride_w;
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = F.max_unpool2d(");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, "kernel_size = (%s,", std::to_string(kernel_size_h).c_str());
                fprintf(pyfp, "%s), ", std::to_string(kernel_size_w).c_str());
                fprintf(pyfp, "stride = (%s,", std::to_string(stride_h).c_str());
                fprintf(pyfp, "%s) ", std::to_string(stride_w).c_str());
                fprintf(pyfp, ")\n");
            }

            else if (op->type.substr(0, 3) == "nn." || op->type.substr(0, 16) == "torchvision.ops.")
            {
                // self.xxx()
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                    if (i + 1 != op->inputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type.find("::") != std::string::npos || op->type.find(".") != std::string::npos)
            {
                // direct
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }

                if (op->type.substr(0, 7) == "Tensor.")
                {
                    if (op->type == "Tensor.fill")
                    {
                        fprintf(pyfp, " = v_%s.fill_(", sanitize_identifier(op->inputs[0]->name).c_str());
                    }
                    else
                    {
                        fprintf(pyfp, " = v_%s.%s(", sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                    }

                    if (op->inputnames.size() == op->inputs.size())
                    {
                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            if (!op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                        }

                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            if (op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "%s=v_%s, ", op->inputnames[i].c_str(), sanitize_identifier(op->inputs[i]->name).c_str());
                        }
                    }
                    else
                    {
                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                        }
                    }
                }
                else
                {
                    //add by senli custom_ops_func
                    if (
                        std::find(custom_ops.begin(), custom_ops.end(), op->type) != custom_ops.end() && customop_infer_py != "None")
                    {
                        std::vector<std::string> op_type_list = split_string(op->type, ".");
                        std::reverse(op_type_list.begin(), op_type_list.end());
                        std::string function_name = op_type_list.at(0);
                        //add by senli 20240320
                        if (std::find(custom_ops_names.begin(), custom_ops_names.end(), function_name) != custom_ops_names.end())
                        {
                            fprintf(pyfp, " = %s(", function_name.c_str());
                        }
                        else
                        {
                            fprintf(pyfp, " = %s(", op->type.c_str());
                        }
                    }
                    else
                    {
                        fprintf(pyfp, " = %s(", op->type.c_str());
                    }

                    if (op->inputnames.size() == op->inputs.size())
                    {
                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            if (!op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }

                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            if (op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "%s=v_%s", op->inputnames[i].c_str(), sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }
                    }
                }

                int i = 0;
                for (const auto& it : op->params)
                {
                    if (op->type.substr(0, 7) == "Tensor." && i == 0)
                    {
                        fprintf(pyfp, "%s=", it.first.c_str());
                    }
                    else if (op->inputs.empty() && i == 0)
                    {
                        fprintf(pyfp, "%s=", it.first.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, ", %s=", it.first.c_str());
                    }

                    i++;

                    const Parameter& param = it.second;
                    if (param.type == 0)
                    {
                        if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            fprintf(pyfp, "torch.tensor(False)");
                        }
                        else
                        {
                            fprintf(pyfp, "None");
                        }
                    }
                    if (param.type == 1)
                    {
                        if (param.b)
                            fprintf(pyfp, "True");
                        else
                            fprintf(pyfp, "False");
                    }
                    if (param.type == 2)
                    {
                        if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            fprintf(pyfp, "torch.tensor(%d)", param.i);
                        }
                        else
                        {
                            fprintf(pyfp, "%d", param.i);
                        }
                    }
                    if (param.type == 3)
                    {
                        if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            fprintf(pyfp, "torch.tensor(%f)", param.f);
                        }
                        else
                        {
                            fprintf(pyfp, "%f", param.f);
                        }
                    }
                    if (param.type == 4)
                    {
                        if (param.s.substr(0, 6) == "torch.")
                        {
                            fprintf(pyfp, "%s", param.s.c_str());
                        }
                        else if (op->type == "Tensor.index_put" && it.first == "values")
                        {
                            if (param.s == "inf" || param.s == "-inf")
                            {
                                fprintf(pyfp, "torch.tensor(float(\'%s\'))", param.s.c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "torch.tensor(\'%s\')", param.s.c_str());
                            }
                        }
                        else
                        {
                            if (param.s == "inf" || param.s == "-inf")
                            {
                                fprintf(pyfp, "float(\'%s\')", param.s.c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "\'%s\'", param.s.c_str());
                            }
                        }
                    }
                    if (param.type == 5)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.ai.size(); i++)
                        {
                            if ((op->type == "F.adaptive_avg_pool2d"
                                    || op->type == "F.adaptive_avg_pool3d"
                                    || op->type == "F.adaptive_max_pool2d"
                                    || op->type == "F.adaptive_max_pool3d")
                                    && it.first == "output_size" && param.ai[i] == 0)
                            {
                                fprintf(pyfp, "None");
                            }
                            else
                            {
                                fprintf(pyfp, "%d", param.ai[i]);
                            }
                            if (i + 1 != param.ai.size() || param.ai.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 6)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.af.size(); i++)
                        {
                            fprintf(pyfp, "%f", param.af[i]);
                            if (i + 1 != param.af.size() || param.af.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 7)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.as.size(); i++)
                        {
                            if (param.as[i].substr(0, 6) == "torch.")
                            {
                                fprintf(pyfp, "%s", param.as[i].c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "\'%s\'", param.as[i].c_str());
                            }
                            if (i + 1 != param.as.size() || param.as.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 10)
                    {
                        fprintf(pyfp, "(%f%+fj)", param.c.real(), param.c.imag());
                    }
                    if (param.type == 11)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.ac.size(); i++)
                        {
                            fprintf(pyfp, "(%f%+fj)", param.ac[i].real(), param.ac[i].imag());
                            if (i + 1 != param.ac.size() || param.ac.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                }

                fprintf(pyfp, ")\n");
            }
            else
            {
                fprintf(stderr, "todo %s\n", op->type.c_str());
            }
        }
    }

    // return add by senli[pnnx_infer]
    fprintf(pyfp, "        if self.infer_flag:\n");
    {
        fprintf(pyfp, "            return ");

        int output_count = 0;
        {
            for (const Operator* op : ops)
            {
                if (op->type == "pnnx.Output")
                    output_count++;
            }
        }

        int output_index = 0;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Output")
                continue;

            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
            if (output_index + 1 != output_count)
                fprintf(pyfp, ", ");

            output_index++;
        }

        fprintf(pyfp, "\n");
    }
    fprintf(pyfp, "        else:\n");

    // return if pre node type is TupleConstruct， max_tensor_index not add one add by senli[pnnx_infer]
    {
        bool TupleConstruct_flag = false;
        int max_tensor_index = 0;
        for (const Operator* op : ops)
        {
            if (op->type == "pnnx.Output")
            {
                std::vector<Operand*> inputs = op->inputs;
                for (const Operand* tensor : inputs)
                {
                    Operator* pre_op = tensor->producer;
                    if (pre_op->type == "prim::TupleConstruct")
                    {
                        TupleConstruct_flag = true;
                    }
                }
                int num = std::stoi(op->inputs[0]->name);
                max_tensor_index = (max_tensor_index > num) ? max_tensor_index : num;
            }
        }
        if (!TupleConstruct_flag)
        {
            max_tensor_index++;
        }
        fprintf(pyfp, "            intermediate = {}\n");
        fprintf(pyfp, "            for i in range(%d):\n", max_tensor_index);
        fprintf(pyfp, "                key = 'v_' + str(i)\n");
        fprintf(pyfp, "                if key in vars().keys():\n");
        fprintf(pyfp, "                    intermediate[str(i)] = vars()[key]\n");
        fprintf(pyfp, "            return intermediate\n");
    }

    fprintf(pyfp, "\n");

    // export torchscript
    {
        fprintf(pyfp, "def export_torchscript():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    mod = torch.jit.trace(net, %s)\n", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    mod = torch.jit.trace(net, (");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, "))\n");
        }

#ifdef _WIN32
        fprintf(pyfp, "    mod.save(r\"%s.pt\")\n", pypath.c_str());
#elif defined(__linux__)
        fprintf(pyfp, "    mod.save(\"%s.pt\")\n", pypath.c_str());
#elif defined(__APPLE__)
        fprintf(pyfp, "    mod.save(\"%s.pt\")\n", pypath.c_str());
#endif
    }

    fprintf(pyfp, "\n");

    // export onnx
    {
        fprintf(pyfp, "def export_onnx():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        // torch.onnx._export(net, v_0, "test_swin_t.onnx", export_params=True, opset_version=14, input_names=['in0'], output_names=['out0'])

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    torch.onnx._export(net, %s", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    torch.onnx._export(net, (");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")");
        }

#ifdef _WIN32
        fprintf(pyfp, ", r\"%s.onnx\", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13", pypath.c_str());
#elif defined(__linux__)
        fprintf(pyfp, ", \"%s.onnx\", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13", pypath.c_str());
#elif defined(__APPLE__)
        fprintf(pyfp, ", \"%s.onnx\", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13", pypath.c_str());
#endif

        fprintf(pyfp, ", input_names=[");
        {
            int input_count = 0;
            {
                for (const Operator* op : ops)
                {
                    if (op->type == "pnnx.Input")
                        input_count++;
                }
            }

            int input_index = 0;
            for (const Operator* op : ops)
            {
                if (op->type != "pnnx.Input")
                    continue;

                fprintf(pyfp, "'in%d'", input_index);
                if (input_index + 1 != input_count)
                    fprintf(pyfp, ", ");

                input_index++;
            }
        }
        fprintf(pyfp, "]");

        fprintf(pyfp, ", output_names=[");
        {
            int output_count = 0;
            {
                for (const Operator* op : ops)
                {
                    if (op->type == "pnnx.Output")
                        output_count++;
                }
            }

            int output_index = 0;
            for (const Operator* op : ops)
            {
                if (op->type != "pnnx.Output")
                    continue;

                fprintf(pyfp, "'out%d'", output_index);
                if (output_index + 1 != output_count)
                    fprintf(pyfp, ", ");

                output_index++;
            }
        }
        fprintf(pyfp, "]");

        fprintf(pyfp, ")\n");
    }

    fprintf(pyfp, "\n");

    // test inference
    //add by senli[pnnx_infer]
    {
        fprintf(pyfp, "def test_inference(bin_path, flag, args):\n");
        fprintf(pyfp, "    net = Model(bin_path,flag)\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "    if isinstance(args, tuple) or isinstance(args, list):\n");
        fprintf(pyfp, "        return net(*args)\n");
        fprintf(pyfp, "    else:\n");
        fprintf(pyfp, "        return net(args)\n");
        // fprintf(pyfp, "    torch.manual_seed(0)\n");

        // std::vector<std::string> input_names;
        // for (const Operator* op : ops)
        // {
        //     if (op->type != "pnnx.Input")
        //         continue;

        //     const Operand* r = op->outputs[0];
        //     std::string input_name = std::string("v_") + sanitize_identifier(r->name);
        //     if (type_is_integer(r->type))
        //     {
        //         fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
        //         for (size_t i = 0; i < r->shape.size(); i++)
        //         {
        //             fprintf(pyfp, "%d", r->shape[i]);
        //             if (i + 1 != r->shape.size() || r->shape.size() == 1)
        //                 fprintf(pyfp, ", ");
        //         }
        //         fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
        //     }
        //     else
        //     {
        //         fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
        //         for (size_t i = 0; i < r->shape.size(); i++)
        //         {
        //             fprintf(pyfp, "%d, ", r->shape[i]);
        //         }
        //         fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
        //     }

        //     input_names.push_back(input_name);
        // }

        // fprintf(pyfp, "\n");

        // if (input_names.size() == 1)
        // {
        //     fprintf(pyfp, "    return net(%s)\n", input_names[0].c_str());
        // }
        // else
        // {
        //     fprintf(pyfp, "    return net(");

        //     for (size_t i = 0; i < input_names.size(); i++)
        //     {
        //         fprintf(pyfp, "%s", input_names[i].c_str());
        //         if (i + 1 != input_names.size())
        //             fprintf(pyfp, ", ");
        //     }

        //     fprintf(pyfp, ")\n");
        // }
    }

    fprintf(pyfp, "\n");

    // main
    {
        fprintf(pyfp, "if __name__ == \"__main__\":\n");
        fprintf(pyfp, "    print(test_inference())\n");
    }

    fclose(pyfp);

    return 0;
}

int Graph::parse(const std::string& param)
{
    std::istringstream is(param);
    if (!is.good())
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int magic = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> magic;
    }

    int operator_count = 0;
    int operand_count = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> operator_count >> operand_count;
    }

    for (int i = 0; i < operator_count; i++)
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int input_count = 0;
        int output_count = 0;

        iss >> type >> name >> input_count >> output_count;

        Operator* op = new_operator(type, name);

        for (int j = 0; j < input_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = get_operand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < output_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = new_operand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
        }

        // key=value
        while (!iss.eof())
        {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);

            if (key[0] == '@')
            {
                // attribute
                //                 load_attribute(op, key.substr(1), value, szr);
                op->attrs[key.substr(1)] = Attribute();

                Attribute& attr = op->attrs[key.substr(1)];

                attr.type = 0;
                if (value.empty())
                    continue;

                if (value[0] == '%')
                {
                    // @data=%op1.data
                    attr.data = std::vector<char>(value.begin(), value.end());
                }

                if (value[0] == '(')
                {
                    // @data=(1,%c,?,4)f32

                    // type
                    std::string typestr = value.substr(value.find_last_of(')') + 1);
                    attr.type = string_to_type(typestr.c_str());

                    // shape
                    std::string lc = value.substr(1, value.find_last_of(')') - 1);
                    std::istringstream lcss(lc);

                    attr.shape.clear();
                    while (!lcss.eof())
                    {
                        std::string elem;
                        std::getline(lcss, elem, ',');

                        if (elem == "?")
                        {
                            attr.shape.push_back(-1);
                        }
                        else if (elem[0] == '%')
                        {
                            // encode %abc as symbolic tag
                            attr.shape.push_back(-233);
                            int index = attr.shape.size() - 1;
                            std::string key = elem.substr(1);
                            attr.params[std::string("__shape_") + std::to_string(index)] = key;
                        }
                        else
                        {
                            int i = std::stoi(elem);
                            attr.shape.push_back(i);
                        }
                    }
                }
            }
            else if (key[0] == '$')
            {
                // operand input key
                load_input_key(op, key.substr(1), value);
            }
            else if (key[0] == '#')
            {
                // operand shape
                load_shape(op, key.substr(1), value);
            }
            else
            {
                // parameter
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

void Operand::remove_consumer(const Operator* c)
{
    auto it = std::find(consumers.begin(), consumers.end(), c);
    if (it != consumers.end())
        consumers.erase(it);
}

Operator* Graph::new_operator(const std::string& type, const std::string& name)
{
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    ops.push_back(op);
    return op;
}

Operator* Graph::new_constant_operator(const std::string& type, const std::string& name)
{
    // get last input index
    int last_input_index = -1;
    for(auto op: this->ops)
    {
        if(op->type == "pnnx.Input")
        {
            last_input_index++;
        }
    }

    if(last_input_index == -1)
    {
        Operator* op = new Operator;
        op->type = type;
        op->name = name;
        ops.push_back(op);
        return op;
    }
    else
    {
        std::string last_input_op_name =  "pnnx_input_" + std::to_string(last_input_index);
        Operator* last_input_op = this->get_operator(last_input_op_name);
        return this->new_operator_after(type, name, last_input_op);
    }
}

Operator* Graph::new_operator_before(const std::string& type, const std::string& name, const Operator* cur)
{
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    ops.insert(std::find(ops.begin(), ops.end(), cur), op);
    return op;
}

Operator* Graph::new_operator_after(const std::string& type, const std::string& name, const Operator* cur)
{
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    ops.insert(std::find(ops.begin(), ops.end(), cur) + 1, op);
    return op;
}

Operand* Graph::new_operand(const std::string& name)
{
    Operand* r = new Operand;
    r->name = name;
    operands.push_back(r);
    return r;
}

Operand* Graph::get_operand(const std::string& name)
{
    for (Operand* r : operands)
    {
        if (r->name == name)
            return r;
    }

    return 0;
}

Operator* Graph::get_operator(const std::string& name)
{
    for (Operator* r : ops)
    {
        if (r->name == name)
            return r;
    }

    return 0;
}
const Operand* Graph::get_operand(const std::string& name) const
{
    for (const Operand* r : operands)
    {
        if (r->name == name)
            return r;
    }

    return 0;
}
int Graph::extract_sub_graph(const std::vector<std::string>& start_nodes, const std::vector<std::string>& end_nodes)
{
    if(start_nodes.size() == 0 && end_nodes.size() == 0)
    {
        fprintf(stderr, "############# not need extract sub graph\n");
    }
    else
    {
        std::vector<std::string> extract_start_nodes = start_nodes;
        std::vector<std::string> extract_end_nodes = end_nodes;
        if(extract_start_nodes.size() == 0)
        {
            // set input node name to start_nodes
            for(auto node :ops)
            {
                if(node->type == "pnnx.Input")
                {
                    std::string input_tensor_name = node->outputs[0]->name;
                    extract_start_nodes.push_back(input_tensor_name);
                }
            }
        }
        if(extract_end_nodes.size() == 0)
        {
            // set output node name to start_nodes
            for(auto node :ops)
            {
                if(node->type == "pnnx.Output")
                {
                    std::string output_tensor_name = node->inputs[0]->name;
                    extract_end_nodes.push_back(output_tensor_name);
                }
            }
        }
        std::vector<Operator*> new_input_ops;
        std::vector<Operator*> new_output_ops;
        //get exclude_node_names exclude_tensor_names
        std::vector<std::string> exclude_node_names;
      
        for(auto node: ops)
        {
            // check is input node or not
            std::vector<Operand*> cur_inputs = node->inputs;
            int input_num = 0;
            for(auto cur_input: cur_inputs)
            {
                std::string cur_node_name = cur_input->name;
                if(std::find(extract_start_nodes.begin(), extract_start_nodes.end(), cur_node_name) != extract_start_nodes.end())
                {
                    bool is_new_tensor = true;
                    for(auto new_op: new_input_ops)
                    {
                        if(new_op->outputs[0]->name == cur_node_name)
                        {
                            is_new_tensor = false;
                            break;
                        }
                    }
                    if(is_new_tensor)
                    {
                        // create new input node 
                        Operator* op = new Operator;
                        op->type = "pnnx.Input";
                        op->name = "pnnx_input_" + std::to_string(new_input_ops.size());
                        op->outputs.push_back(cur_input);
                        std::vector<int> shape = cur_input->shape;
                        
                        new_input_ops.push_back(op);

                        // get pre node 
                        Operator* pre_node = cur_input->producer;
                        if(std::find(exclude_node_names.begin(), exclude_node_names.end(), pre_node->name) == exclude_node_names.end())
                        {
                            exclude_node_names.push_back(pre_node->name);
                            std::list<Operator*> List; 
                            List.push_back(pre_node);
                            while(!List.empty())
                            {
                                Operator* cur_node = List.front();
                                List.pop_front();
                                std::vector<Operand*> cur_node_inputs = cur_node->inputs;
                                for(auto cur_node_input: cur_node_inputs)
                                {
                                    Operator* pre_node_producer = cur_node_input->producer;
                                    if(std::find(exclude_node_names.begin(), exclude_node_names.end(), pre_node_producer->name) == exclude_node_names.end())
                                    {
                                        exclude_node_names.push_back(pre_node_producer->name);
                                        List.push_back(pre_node_producer);
                                    }
                        
                                }

                            } 
                        }
                        
                        
                    }

                    input_num++;
                }
   
            }
            // is start node
            if(input_num != 0)
            {
                if(input_num > extract_start_nodes.size())
                {
                    fprintf(stderr, "############# please check your start nodes!\n");
                    return -1;
                }
            }

             // check is output node or not
            std::vector<Operand*> cur_outputs = node->outputs;
            int output_num = 0;
            for(auto cur_output: cur_outputs)
            {
                std::string cur_node_name = cur_output->name;
                if(std::find(extract_end_nodes.begin(), extract_end_nodes.end(), cur_node_name) != extract_end_nodes.end())
                {
                    bool is_new_tensor = true;
                    for(auto new_op: new_output_ops)
                    {
                        if(new_op->inputs[0]->name == cur_node_name)
                        {
                            is_new_tensor = false;
                            break;
                        }
                    }
                    if(is_new_tensor)
                    {
                        Operator* op = new Operator;
                        op->type = "pnnx.Output";
                        op->name = "pnnx_Output_" + std::to_string(new_output_ops.size());
                        op->inputs.push_back(cur_output);
                        std::vector<int> shape = cur_output->shape;
                        
                        new_output_ops.push_back(op);


                         // get sink node 
                        std::vector<Operator*> sink_nodes = cur_output->consumers;
                        for(auto sink_node: sink_nodes)
                        {
                            if(std::find(exclude_node_names.begin(), exclude_node_names.end(), sink_node->name) == exclude_node_names.end())
                            {
                                exclude_node_names.push_back(sink_node->name);
                                std::list<Operator*> sink_List; 
                                sink_List.push_back(sink_node);
                                while(!sink_List.empty())
                                {
                                    Operator* cur_sink_node = sink_List.front();
                                    sink_List.pop_front();
                                    std::vector<Operand*> cur_sink_node_outputs = cur_sink_node->outputs;
                                    for(auto cur_sink_node_output: cur_sink_node_outputs)
                                    {
                                        std::vector<Operator*> sink_node_consumers = cur_sink_node_output->consumers;
                                        for(auto sink_node_consumer: sink_node_consumers)
                                        {
                                            if(std::find(exclude_node_names.begin(), exclude_node_names.end(), sink_node_consumer->name) == exclude_node_names.end())
                                            {
                                                exclude_node_names.push_back(sink_node_consumer->name);
                                                sink_List.push_back(sink_node_consumer);
                                            }
                                        }
                                        
                                    }

                                } 
                            }
                        }
                        
                    }

                    output_num++;
                }
   
            }

            // is end node
            if(output_num != 0)
            {
                if(output_num > extract_end_nodes.size())
                {
                    fprintf(stderr, "############# please check your end nodes!\n");
                    return -1;
                }
            }  
        }

        // delect exclude_node_names
        while (1)
        {
            bool matched = false;

            for (size_t i = 0; i < ops.size(); i++)
            {
                Operator* op = ops[i];
                if(std::find(exclude_node_names.begin(), exclude_node_names.end(),op->name) == exclude_node_names.end())
                {
                    continue;
                }
                matched = true;
                std::vector<Operand*> inputs = op->inputs;
                std::vector<Operand*> outputs = op->outputs;
                for(auto match_node_output: outputs)
                {
                    if(std::find(extract_start_nodes.begin(), extract_start_nodes.end(), match_node_output->name) != extract_start_nodes.end())
                    {
                        for(auto new_input_op: new_input_ops)
                        {
                            for(auto new_input_op_output: new_input_op->outputs)
                            {
                                if(new_input_op_output->name == match_node_output->name)
                                {
                                    match_node_output->producer = new_input_op;
                                }
                            }
                        }
                    }
                    else
                    {
                        match_node_output->producer = 0;
                        match_node_output->consumers.clear();
                        if(std::find(operands.begin(), operands.end(), match_node_output) != operands.end())
                        {
                            operands.erase(std::find(operands.begin(), operands.end(), match_node_output));
                            delete match_node_output;
                        }
                        
                    }
                }

                for(auto match_node_input: inputs)
                {
                    if(std::find(extract_end_nodes.begin(), extract_end_nodes.end(), match_node_input->name) != extract_end_nodes.end())
                    {
                        for(auto new_output_op: new_output_ops)
                        {
                            for(auto new_output_op_input: new_output_op->inputs)
                            {
                                if(new_output_op_input->name == match_node_input->name)
                                {
                                    match_node_input->consumers.push_back(new_output_op);
                                }
                            }
                        }
                    }
                    else
                    {
                        match_node_input->producer = 0;
                        match_node_input->consumers.clear();
                        if(std::find(operands.begin(), operands.end(), match_node_input) != operands.end())
                        {
                            operands.erase(std::find(operands.begin(), operands.end(), match_node_input));
                            delete match_node_input;
                        }

                    }
                }
                op->inputs.clear();
                op->outputs.clear();

                ops.erase(ops.begin() + i);
                delete op;
                break;
            }

            if (!matched)
            break;
        }

        // insert new input outout node
        ops.insert(ops.end(), new_input_ops.begin(), new_input_ops.end()); 
        ops.insert(ops.end(), new_output_ops.begin(), new_output_ops.end());
        
    }
    return 1; 
}


std::string MainGraph::get_pnnx_graph_name()
{
    return name;
}
void MainGraph::create_main_graph(std::string& name)
{
    this->name = name;
    this->main_graph = std::make_shared<Graph>();
}

std::shared_ptr<pnnx::Graph> MainGraph::get_main_graph()
{
    return this->main_graph;
} 

void MainGraph::insert_sub_graph(std::string& name, std::shared_ptr<pnnx::MainGraph>& sub_graph, Operator* op, int init_input_num)
{
    this->sub_graph_map[name] = sub_graph;
    std::vector<int> init_index = {};
    for(auto i = 0; i < init_input_num; i++)
    {
        init_index.push_back(i);
    }
    std::unordered_map<std::string, std::vector<int>> op_graph_input_indexes = {{name, init_index}};
    op_2_graph[op->name] = op_graph_input_indexes;
}

void MainGraph::set_base_graph(std::shared_ptr<pnnx::MainGraph>& base_graph)
{
    this->base_graph = base_graph;
}

std::shared_ptr<pnnx::MainGraph> MainGraph::get_base_graph()
{
    return this->base_graph;
}

std::shared_ptr<pnnx::MainGraph> MainGraph::get_sub_graph(std::string& name)
{
    return  this->sub_graph_map[name];
}
Operator* MainGraph::set_sub_graph_new_input(const std::string& sub_graph_name, const std::string& operand_name, Operand* r1)
{
    auto sub_graph = this->sub_graph_map[sub_graph_name];
    auto sub_main_graph = sub_graph->get_main_graph();
    // create new input
    int input_index = 0;
    for(auto op: sub_main_graph->ops)
    {
        if(op->type == "pnnx.Input")
        {
            input_index++;
        }
    }
    Operator* new_input_op;
    std::string new_input_op_name =  "pnnx_input_" + std::to_string(input_index);
    if(input_index == 0)
    {
        new_input_op = sub_main_graph->new_operator("pnnx.Input", new_input_op_name);
    }
    else
    {
        std::string last_input_op_name =  "pnnx_input_" + std::to_string(input_index-1);
        Operator* last_input_op = sub_main_graph->get_operator(last_input_op_name);
        new_input_op = sub_main_graph->new_operator_after("pnnx.Input", new_input_op_name, last_input_op);
        
    }
    Operand* r2 = sub_main_graph->new_operand(operand_name);
    r2->producer = new_input_op;
    r2->params = r1->params;
    r2->type = r1->type;
    r2->shape = r1->shape;
    new_input_op->outputs.push_back(r2);
    return new_input_op;
}

void MainGraph::set_op_new_input(std::string& sub_graph_name2,  Operator* new_input_op)
{
    for (auto it = op_2_graph.begin(); it != op_2_graph.end(); ++it) 
    {  
        auto& op_2_graph_input_list = it->second;
        for (auto it2 = op_2_graph_input_list.begin(); it2 != op_2_graph_input_list.end(); ++it2) 
        {
            if(it2->first == sub_graph_name2)
            {
                // get op 
                
                Operator* src_op = main_graph->get_operator(it->first);
                
                new_input_op->outputs[0]->consumers.push_back(src_op);
                src_op->inputs.push_back(new_input_op->outputs[0]);
                int last_src_input_index = src_op->inputs.size() -1; 
                it2->second.push_back(last_src_input_index);
            }
        }  
    }
}

Operator* MainGraph::get_base_op(std::string& sub_op_block_name)
{
   auto base_main_graph =  base_graph->get_main_graph();
   for(auto op: base_main_graph->ops)
   {
        if(op->name == sub_op_block_name) 
        {
            return op;
        }
   }
   return 0;
}

MainGraph::MainGraph()
{
}


MainGraph::~MainGraph()
{

}

MainGraph::MainGraph(const MainGraph& /*rhs*/)
{
}

MainGraph& MainGraph::operator=(const MainGraph& /*rhs*/)
{
    return *this;
}





} // namespace pnnx
