# Faster HO-RCNN

The code is based around https://github.com/jinfagang/keras_frcnn,  https://github.com/broadinstitute/keras-rcnn and https://github.com/endernewton/tf-faster-rcnn with additional parts from https://github.com/rbgirshick/py-faster-rcnn and https://github.com/rbgirshick/fast-rcnn and the official HO-RCNN Caffe implementation https://github.com/ywchao/ho-rcnn.


Training HOI detection

(a) Data has to be downloaded and converted to the correct datatype.

(b-1) All models are trained running the training scripts in detection/scripts/

(b-2) Before each model is trained the inputs be created and saved by running the save-input scripts in detection/scripts/

(c) Train all sub-models again with shared CNN running the same scripts but with different arguments



Testing HOI detection

(a) Data has to be downloaded and converted to the correct datatype)

(b) The data is fed through all sub-models in the test-all-models script in detection/tests/



Training and testing HOI classification with ground truth bounding boxes

(a) Data has to be downloaded and converted to the correct datatype)

(b) Train the model with the rcnn script in classification/scripts/

(c) Evaluate the model with the eval script in classification/scripts/

