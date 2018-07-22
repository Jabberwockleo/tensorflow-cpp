DST="/tmp/tf"
echo "===copying src"
cp -RL ./tensorflow/core $DST/tensorflow
cp -RL ./tensorflow/cc $DST/tensorflow

echo "===copying third"
mkdir -p $DST/third_party
cp -RL ./third_party/eigen3 $DST/third_party

echo "===copying external"
#rm -rf $DST/third_party/eigen3/unsupported
cp -RLf ./bazel-tensorflow/external/eigen_archive/unsupported $DST

echo "===copying genfiles"
cp -RL ./bazel-genfiles/tensorflow/cc $DST/tensorflow
cp -RL ./bazel-genfiles/tensorflow/core $DST/tensorflow

echo "===copying archive"
cp -RL ./bazel-tensorflow/external/eigen_archive/Eigen $DST/Eigen