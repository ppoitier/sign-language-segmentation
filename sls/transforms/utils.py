from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transform import *


def get_transform_pipeline(pipeline_name: str):
    if pipeline_name == 'optical-flow':
        return Compose([
            Concatenate(["upper_pose", "left_hand", "right_hand"]),
            ToOpticalFlow(fps=50),
            # Padding(min_length=1500, mode='edge'),
        ])
    raise ValueError(f"Unknown transform pipeline: {pipeline_name}")
