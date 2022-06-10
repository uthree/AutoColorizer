unet_configs = {
        "style": True,
        "style_dim": 256,
        "input_channels": 1,
        "output_channels": 3,
        "channels": [32, 64, 128, 256],
        "stages": [3, 3, 9, 3],
        }
style_encoder_configs = {
        "output_features": 256,
        "channels": [16, 32, 64, 128],
        "stages": [2, 2, 2, 2],
        }
discriminator_configs = {
        "output_features": 1,
        "channels": [24, 32, 64, 128],
        "stages": [3, 3, 3, 3],
        "minibatch_std": True,
        }
