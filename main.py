import os 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.tsne import tsne
from utils.utils import load_species_split, predictions_last_epochs,save_config,report,save_log
from utils.plot import families_plot, plot_loss, pression_recall, prob_distribution, species_plot
from data.utils import species_name_extraction, extract_labels, extract_labels_and_image, startup_dir
from data.dataset import ImageDataset
from data.health import check
from model.model_import import classifier
from train.losses import get_loss_function
from train.scheduler import get_optimizer, get_scheduler
from train.trainer import train
from train.metrics import MultiLabelMetricsSaver

def main(opt):

    destination_dir = startup_dir(opt.name)

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        dest = destination_dir.split("/")[-1]
        logger = SummaryWriter(os.path.join("runs", dest))

    save_config(vars(opt),destination_dir)

    unique_species_name_list, all_species_info = species_name_extraction(opt.img_limitation,opt.path_source_img)
    train_species,valid_species = load_species_split(opt.path_source_split)

    labels_in_csv, families, hab_names, habitats = extract_labels(opt.path_source_csv,unique_species_name_list)
    (list_of_train_image_labels, list_of_train_image_path, list_of_train_image_info), (list_of_valid_image_labels, list_of_valid_image_path, list_of_valid_image_info), list_of_family_id, list_of_species_id = extract_labels_and_image(all_species_info, labels_in_csv, families, (train_species,valid_species))

    check((list_of_train_image_labels, list_of_train_image_info),(list_of_valid_image_labels, list_of_valid_image_info),hab_names,destination_dir)


    train_dataset = ImageDataset((opt.img_size,opt.path_source_img,opt.model),list_of_train_image_path, list_of_train_image_labels, list_of_train_image_info, batch_size=opt.train_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    valid_dataset = ImageDataset((opt.img_size,opt.path_source_img,opt.model),list_of_valid_image_path, list_of_valid_image_labels, list_of_valid_image_info, batch_size=opt.valid_batch_size, valid=True)
    valid_loader = DataLoader(valid_dataset, batch_size=None, shuffle=False)

    model = classifier(opt.model,output_dim=np.array(list_of_train_image_labels).shape[1])
    device = torch.device(opt.device)

    model.to(device)

    criterion = get_loss_function(opt.loss)
    optimizer = get_optimizer(opt.opt,opt.lr,opt.weight_decay,model)
    scheduler = get_scheduler(opt.warmup_epochs,opt.epochs,optimizer)

    if opt.tensorboard:
        logger = SummaryWriter(os.path.join("runs", destination_dir.split("/")[-1]))

    trained_model, metrics, complete_log = train(opt.epochs, opt.device,(train_loader,valid_loader),model,criterion,optimizer,scheduler,MultiLabelMetricsSaver,np.array(list_of_train_image_labels).shape[1],logger if opt.tensorboard else None)

    save_log(complete_log,destination_dir)
    torch.save(trained_model.state_dict(),os.path.join(destination_dir,"model"))  


    train_dataset = ImageDataset((opt.img_size,opt.path_source_img,opt.model),list_of_train_image_path, list_of_train_image_labels, list_of_train_image_info, batch_size=opt.valid_batch_size, valid=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    (all_train_preds,all_train_labels,all_train_features,all_train_real_prob,all_train_species,all_train_sexe,all_train_family), (all_valid_preds,all_valid_labels,all_valid_features,all_valid_real_prob,all_valid_species,all_valid_sexe,all_valid_family) = predictions_last_epochs(opt.device,model,(train_loader,valid_loader))


    report(all_valid_labels, all_valid_preds,destination_dir)

    plot_loss(opt.epochs,metrics,destination_dir)

    pression_recall(all_valid_real_prob,all_valid_labels,habitats,destination_dir)
    families_plot(all_valid_real_prob,all_valid_labels,all_valid_family,habitats,families,destination_dir)
    species_plot(all_valid_real_prob,all_valid_labels,all_valid_species,destination_dir)
    prob_distribution(all_valid_real_prob,all_valid_labels,habitats,destination_dir)

    tsne(all_valid_features,all_valid_labels,destination_dir)



def parse_opt():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", "-n", type=str, required=True, help="Name of dir")
    parser.add_argument("--model","-m", type=str, choices=["dinov2_vitl14_reg", "dinov2_vitl14", "inceptionv4","dinov2_vitl14Scratch", "eva02L"], default="dinov2_vitl14", help="model type")
    parser.add_argument("--loss", "-l", type=str, default="bcewithlogitsloss", help="Name of dir")
    parser.add_argument("--train_batch_size", "-tbs", type=int, default=40, help="train batch size")
    parser.add_argument("--valid_batch_size", "-vbs", type=int, default=40, help="batch size")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--path_source_csv","-psc", type=str, default="/home/oriol@newcefe.newage.fr/Datasets/final_table2.csv", help="path to csv file")
    parser.add_argument("--path_source_img","-psi", type=str, default="/home/oriol@newcefe.newage.fr/Datasets//whole_bird", help="path to image folder")
    parser.add_argument("--path_source_split","-pss", type=str, default="/home/oriol@newcefe.newage.fr/Models/project_multilabel/saved_mult_split_limit:1000.pkl", help="path to split file")
    parser.add_argument("--img_size", "-size", type=int, default=224, help="image size")
    parser.add_argument("--img_limitation","-iml", type=int, default=None, help="image limitation (None for no limitation)")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="number of epochs")
    parser.add_argument("--opt", "-o", type=str, default="adam", help="optimizer type")
    parser.add_argument("--lr", "-lr", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--warmup_epochs", "-we", type=int, default=5, help="number of warmup epochs")  
    parser.add_argument("--tensorboard", "-board", type=bool, default=False, help="Tensorboard logging")  
    return parser.parse_args()
                        
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)