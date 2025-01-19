from sign_language_tools.common.transforms import (
    Compose, TransformTuple, Concatenate as ConcatenateChannels, MapTransform, Identity
)
from sign_language_tools.pose.transform import *


def get_transform_pipeline(pipeline_name: str):
    if pipeline_name == 'optical-flow':
        return Compose([
            Concatenate(["upper_pose", "left_hand", "right_hand"]),
            ToOpticalFlow(fps=50),
        ])
    elif pipeline_name == 'norm+optical-flow':
        return Compose([
            Concatenate(["upper_pose", "left_hand", "right_hand"]),
            NormalizeEdgeLengths(unitary_edge=(11, 12)),
            CenterOnLandmarks((11, 12)),
            ToOpticalFlow(fps=50),
        ])
    elif pipeline_name == 'flatten-pose':
        return Compose([
            Concatenate(["upper_pose", "left_hand", "right_hand"]),
            DropCoordinates('z'),
            Flatten(),
        ])
    elif pipeline_name == 'norm+flatten-pose':
        return Compose([
            Concatenate(["upper_pose", "left_hand", "right_hand"]),
            DropCoordinates('z'),
            NormalizeEdgeLengths(unitary_edge=(11, 12)),
            CenterOnLandmarks((11, 12)),
            Flatten(),
        ])
    elif '||' in pipeline_name:
        pipeline_names = pipeline_name.split('||')
        pipelines = [get_transform_pipeline(name) for name in pipeline_names]
        return Compose([
            TransformTuple(Identity(), n=len(pipelines)),
            MapTransform(pipelines),
            ConcatenateChannels(dim=-1),
        ])
    raise ValueError(f"Unknown transform pipeline: {pipeline_name}")
