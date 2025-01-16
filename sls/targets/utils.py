from .decoders import ActionnessDecoder, BIOTagDecoder, OffsetsDecoder
from .encoders import ActionnessEncoder, BIOTagEncoder, OffsetsWithSegmentationEncoder


def get_target_encoder(encoder_name: str, encoder_args: dict, length: int):
    encoder_args["length"] = length
    if encoder_name == "actionness":
        return ActionnessEncoder(**encoder_args)
    elif encoder_name == "bio_tags":
        return BIOTagEncoder(**encoder_args)
    elif encoder_name == "offsets+bio_tags":
        return OffsetsWithSegmentationEncoder(
            length=encoder_args["length"],
            segmentation_encoder=BIOTagEncoder(**encoder_args),
        )
    elif encoder_name == "offsets+actionness":
        return OffsetsWithSegmentationEncoder(
            length=encoder_args["length"],
            segmentation_encoder=ActionnessEncoder(**encoder_args),
        )
    else:
        raise ValueError(f"Unknown target encoder: {encoder_name}")


def get_target_decoder(decoder_name: str, decoder_args: dict):
    if decoder_name == "actionness":
        return ActionnessDecoder()
    elif decoder_name == "bio_tags":
        return BIOTagDecoder()
    elif decoder_name == 'offsets+actionness':
        return OffsetsDecoder(**decoder_args)
    else:
        raise ValueError(f"Unknown target decoder: {decoder_name}")
