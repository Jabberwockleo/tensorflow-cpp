/***************************************************************************
 * 
 * Copyright (c) 2018 Wan Li. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file ut/generalized.cpp
 * @author Wan Li
 * @date 2018/07/22 14:53:06
 * @version $Revision$ 
 * @brief 
 *  
 **/
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

int main(int argc, char **argv) {
    using namespace tensorflow;

    const std::string export_dir = "../data/saved/";

    SavedModelBundle bundle;
    SessionOptions session_options;
    RunOptions run_options;
    auto status = LoadSavedModel(session_options, run_options, export_dir,
            {tensorflow::kSavedModelTagServe}, &bundle);
    if (status.ok()) {
        std::cout << "loaded." << std::endl;
    } else {
        std::cout << "load failed." << std::endl;
        return -1;
    }

    auto &meta_graph_def = bundle.meta_graph_def;
    auto &session = bundle.session;

    const auto &signature_def = meta_graph_def.signature_def()["model"];
    const auto &tensor_info_in = signature_def.inputs()["x"];
    const auto &tensor_info_out = signature_def.outputs()["y"];

    std::vector<std::pair<std::string, Tensor> > inputs_conf;
    std::vector<std::string> outputs_conf;
    std::vector<std::string> targets_conf;

    Scope root = Scope::NewRootScope();
    Tensor in_0(tensorflow::DT_FLOAT, TensorShape({1, 1}));
    auto t_matrix = in_0.matrix<float>();
    t_matrix(0, 0) = 2.f;
    inputs_conf.emplace_back(std::make_pair(tensor_info_in.name(), in_0));
    outputs_conf.emplace_back(tensor_info_out.name());

    std::vector<Tensor> outputs;
    TF_CHECK_OK(session->Run(inputs_conf,
                outputs_conf,
                targets_conf,
                &outputs));
    std::cout << outputs[0].matrix<float>() << std::endl;

    return 0;
}
/* vim: set ts=4 sw=4 sts=4 tw=100 */
