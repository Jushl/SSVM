from models.SMVM.model import SMVMDetectionModel


def freeze_streaming_layers(model, k=".m1"):
    for name, param in model.named_parameters():
        if k in name:
            param.requires_grad = False
    print(f"Freeze streaming block")


def unfreeze_streaming_layers(model, k=".m1"):
    for name, param in model.named_parameters():
        if k in name:
            param.requires_grad = True
    print(f"Unfreeze streaming block")


def build_SMVM_Detector(args):
    model = SMVMDetectionModel(args)
    freeze_streaming_layers(model, k=".m1")
    return model
