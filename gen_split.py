import pickle
import numpy as np 

from data.utils import extract_labels, species_name_extraction

def extract_labels_and_image(all_species_name, labels_in_csv):

    list_of_spe_labels = []
    list_of_spe = []
    save = {}
    for species in all_species_name.keys():
            if species not in labels_in_csv: continue # Certains espèces ne se retrouve pas dans le csv
            if species in save : continue
            else : 
                save[species]=[]
            list_of_spe.append(species)
            list_of_spe_labels.append(labels_in_csv[species])

    return list_of_spe, list_of_spe_labels

def main(opt):

    unique_species_name_list, all_species_info = species_name_extraction(opt.img_limitation,opt.path_source_img)
    labels_in_csv, families, hab_names, habitats = extract_labels(opt.path_source_csv,unique_species_name_list)
    list_of_spe, list_of_spe_labels = extract_labels_and_image(all_species_info, labels_in_csv)

    X = np.array(list_of_spe)
    y = np.array(list_of_spe_labels)

    from skmultilearn.model_selection import IterativeStratification

    stratifier = IterativeStratification(n_splits=2, order=1)
    for train_idx, val_idx in stratifier.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]


    with open(f"saved_mult_split_limit:{opt.img_limitation}.pkl", "wb") as f:
        pickle.dump({"train_species": X_train, "valid_species": X_val}, f)



def parse_opt():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_source_csv","-psc", type=str, default="/home/oriol@newcefe.newage.fr/Datasets/final_table2.csv", help="path to csv file")
    parser.add_argument("--path_source_img","-psi", type=str, default="/home/oriol@newcefe.newage.fr/Datasets/whole_bird", help="path to image folder")
    parser.add_argument("--img_limitation","-iml", type=int, default=None, help="image limitation (None for no limitation)")

    return parser.parse_args()
                        
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)