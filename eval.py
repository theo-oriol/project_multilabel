import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import ImageDataset
from data.health import check
from data.utils import extract_labels, extract_labels_and_image, species_name_extraction, startup_dir
from model.model_import import classifier
from utils.tsne import tsne
from utils.plot import f1_per_cls, prob_distribution, families_plot, pression_recall, species_plot
from utils.utils import load_species_split, predictions_last_epochs, report

def eval(opt):
    
    destination_dir = startup_dir(opt.name)

    

    unique_species_name_list, all_species_info = species_name_extraction(opt.img_limitation,opt.path_source_img)
    train_species,valid_species = load_species_split(opt.path_source_split)

    labels_in_csv, families, hab_names, habitats = extract_labels(opt.path_source_csv,unique_species_name_list)
    (list_of_train_image_labels, list_of_train_image_path, list_of_train_image_info), (list_of_valid_image_labels, list_of_valid_image_path, list_of_valid_image_info), list_of_family_id, list_of_species_id = extract_labels_and_image(all_species_info, labels_in_csv, families, (train_species,valid_species))

    check((list_of_train_image_labels, list_of_train_image_info),(list_of_valid_image_labels, list_of_valid_image_info),hab_names,destination_dir)

    valid_dataset = ImageDataset((opt.img_size,opt.path_source_img,opt.model),list_of_valid_image_path, list_of_valid_image_labels, list_of_valid_image_info, batch_size=opt.valid_batch_size, valid=True)
    valid_loader = DataLoader(valid_dataset, batch_size=None, shuffle=False)

    model = classifier(opt.model,output_dim=np.array(list_of_train_image_labels).shape[1])
    model.load_state_dict(torch.load(opt.path_model, weights_only=True))
    model.eval()
    device = torch.device(opt.device)
    model.to(device)

    (all_valid_preds,all_valid_labels,all_valid_features,all_valid_real_prob,all_valid_species,all_valid_sexe,all_valid_family) = predictions_last_epochs(opt.device,model,((),valid_loader),justvalid=True)

    report(all_valid_labels, all_valid_preds,destination_dir)
    pression_recall(all_valid_real_prob,all_valid_labels,habitats,destination_dir)
    families_plot(all_valid_real_prob,all_valid_labels,all_valid_family,habitats,families,destination_dir)
    species_plot(all_valid_real_prob,all_valid_labels,all_valid_species,destination_dir)
    prob_distribution(all_valid_real_prob,all_valid_labels,habitats,destination_dir)
    f1_per_cls(all_valid_real_prob,all_valid_labels,habitats,destination_dir)
    tsne(all_valid_features,all_valid_labels,destination_dir)

def parse_opt():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, required=True, help="Name of dir")
    parser.add_argument("--model","-m", type=str, choices=["dinov2_vitl14_reg", "dinov2_vitl14", "inceptionv4","dinov2_vitl14Scratch", "eva02L"], default="dinov2_vitl14", help="model type")
    parser.add_argument("--valid_batch_size", "-b", type=int, default=40, help="batch size")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--path_model", "-pm", type=str, required=True, help="model weights path")
    parser.add_argument("--path_source_csv","-psc", type=str, default="/home/oriol@newcefe.newage.fr/Datasets/final_table2.csv", help="path to csv file")
    parser.add_argument("--path_source_img","-psi", type=str, default="/home/oriol@newcefe.newage.fr/Datasets//whole_bird", help="path to image folder")
    parser.add_argument("--path_source_split","-pss", type=str, default="/home/oriol@newcefe.newage.fr/Models/project/saved_split_limit:None.pkl", help="path to split file")
    parser.add_argument("--img_size", "-size", type=int, default=224, help="image size")
    parser.add_argument("--img_limitation","-iml", type=int, default=None, help="image limitation (None for no limitation)")
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    eval(opt)

    