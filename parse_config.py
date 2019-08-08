from google.protobuf import json_format,text_format
from object_detection.protos import pipeline_pb2

import tensorflow as tf
import json,os



def protobufConvertJsonFile(protobufFilePath,JsonFilePath):

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(protobufFilePath, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)


    serialized =json_format.MessageToJson(pipeline_config)
    obj = json.loads(serialized)


    #json file save with python
    with open(JsonFilePath, 'w') as f:
        json.dump(obj, f)



if __name__ == '__main__':
    mainPath = os.getcwd()
    protobufFilePath = os.path.join(mainPath,"rfcn_resnet101_coco.config")
    JsonFilePath = os.path.join(mainPath,"rfcn_resnet101_coco.json")
    protobufConvertJsonFile(protobufFilePath,JsonFilePath)




