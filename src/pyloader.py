#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: pyloader.py
Author: Wan Li
Date: 2018/07/22 14:44:18
"""

import tensorflow as tf

if __name__ == "__main__":
    export_dir = "../data/saved/"
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        signature = meta_graph_def.signature_def
        x_tensor_name = signature["model"].inputs["x"].name
        y_tensor_name = signature["model"].outputs["y"].name
        print("x_tensor_name:", x_tensor_name)
        print("y_tensor_name:", y_tensor_name)
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        model = sess.graph.get_tensor_by_name(y_tensor_name)
        print(sess.run(model, feed_dict={x: [2]}))
