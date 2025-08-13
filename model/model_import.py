def classifier(m, output_dim):

    if m == "dinov2_vitl14_reg":
        from model.dino_model import dinov2_vitl14_reg as Classifier
        return Classifier(output_dim=output_dim)
    elif m == "dinov2_vitl14":
        from model.dino_model import dinov2_vitl14 as Classifier
        return Classifier(output_dim=output_dim)
    elif m == "inceptionv4":
        from model.inception import inceptionv4 as Classifier
        return Classifier(output_dim=output_dim)
    elif m == "dinov2_vitl14Scratch":
        from model.dino_model import dinov2_vitl14Scratch as Classifier
        return Classifier(output_dim=output_dim)
    elif m == "eva02L":
        from model.eva import eva02L as Classifier
        return Classifier()
    else : raise ValueError(f"Unknown model type: {m}")
    