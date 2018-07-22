#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: pysaver.py
Author: Wan Li
Date: 2018/07/22 14:47:52
"""

import tensorflow as tf

if __name__ == "__main__":
    export_dir = "../data/saved/"
    # save
    tf.reset_default_graph()
    vi = tf.placeholder(tf.float32, shape=[1])
    v1 = tf.get_variable("v1", shape=[1], initializer = tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[1], initializer = tf.zeros_initializer)
    vo = vi * (v1 - v2)
    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        inc_v1.op.run()
        dec_v2.op.run()
        print(sess.run(vo, feed_dict={vi: [2]}))
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map= {
            "model": tf.saved_model.signature_def_utils.build_signature_def(
                inputs= {"x": tf.saved_model.utils.build_tensor_info(vi)},
                outputs= {"y": tf.saved_model.utils.build_tensor_info(vo)})
        })
        builder.save()
