from .encoders import ActionnessEncoder, BIOTagEncoder, BoundariesEncoder
from .decoders import ActionnessDecoder


def get_target_encoder(target: str, length: int, **kwargs):
    if target == 'actionness':
        return ActionnessEncoder(length)
    elif target == 'bio_tags':
        return BIOTagEncoder(length, **kwargs)
    elif target == 'thin_boundaries':
        return BoundariesEncoder(length, **kwargs)
    else:
        raise ValueError(f"No encoder found for target: {target}")


def get_target_decoder(target: str):
    if target == 'actionness':
        return ActionnessDecoder()
    else:
        raise ValueError(f"No decoder found for target: {target}")
