import timm
def build_convnextv2(config, img_size, pt_root):
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384',
                               pretrained=True,
                               features_only=True,
                               out_indices=(0, 1, 2, 3))
    embed_dims = [128, 256, 512, 1024]  # Base
    return model, embed_dims
