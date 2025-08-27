import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def check(train,valid, habitat_name,destination_dir):
    (list_of_train_image_labels, list_of_train_image_info) = train 
    (list_of_valid_image_labels, list_of_valid_image_info) = valid

    list_of_train_image_sexe, list_of_valid_image_sexe = [],[]
    for i in range(len(list_of_train_image_info)):
        list_of_train_image_sexe.append(list_of_train_image_info[i][1])
    for i in range(len(list_of_valid_image_info)):
        list_of_valid_image_sexe.append(list_of_valid_image_info[i][1])

    list_of_image_species = []
    for i in range(len(list_of_train_image_info)):
        list_of_image_species.append(list_of_train_image_info[i][0])
    for i in range(len(list_of_valid_image_info)):
        list_of_image_species.append(list_of_valid_image_info[i][0])

    list_of_image_family = []
    for i in range(len(list_of_train_image_info)):
        list_of_image_family.append(list_of_train_image_info[i][2])
    for i in range(len(list_of_valid_image_info)):
        list_of_image_family.append(list_of_valid_image_info[i][2])

    plot_species_balance(list_of_image_species, destination_dir)
    plot_family_balance(list_of_image_family, destination_dir)
    plot_class_distribution(habitat_name, (list_of_train_image_labels,list_of_valid_image_labels), destination_dir)
    plot_sexe_distribution((list_of_train_image_sexe, list_of_valid_image_sexe), destination_dir)



def plot_species_balance(species, destination_dir):

    species_counts = np.zeros(np.max(species)+1)
    for s in species:
        species_counts[s] += 1


    plt.figure()
    plt.hist(species_counts, bins=4, edgecolor='black')
    plt.ylabel('Number of Samples')
    plt.title('Species Balance')
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir, "Species_balance"))
    plt.close()

def plot_family_balance(families, destination_dir):
    family_counts = np.zeros(np.max(families)+1)
    for f in families:
        family_counts[f] += 1

    plt.figure(figsize=(12, 8))
    plt.hist(family_counts, bins=20, edgecolor='black')
    plt.xlabel('Number of Samples per Family')
    plt.ylabel('Frequency')
    plt.title('Family Sample Count Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir, "Family_distribution"))


def plot_class_distribution(habitats,labels,destination_dir):
    all_train_labels, all_valid_labels = labels
    num_classes = np.array(all_train_labels).shape[1]  

    train_class_counts = np.sum(all_train_labels, axis=0)
    valid_class_counts = np.sum(all_valid_labels, axis=0)
    total_class_counts = train_class_counts + valid_class_counts

    x = np.arange(num_classes) 
    width = 0.25  

    total_train = train_class_counts.sum()
    total_valid = valid_class_counts.sum()
    total_all   = total_class_counts.sum()

    plt.figure(figsize=(14, 6))
    
    bars_train = plt.bar(x - width, train_class_counts, width, label='Train')
    bars_valid = plt.bar(x, valid_class_counts, width, label='Validation')
    bars_total = plt.bar(x + width, total_class_counts, width, label='Total')

    for idx, (bar, count) in enumerate(zip(bars_train, train_class_counts)):
        pct = 100 * count / total_train if total_train else 0
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)

    for idx, (bar, count) in enumerate(zip(bars_valid, valid_class_counts)):
        pct = 100 * count / total_valid if total_valid else 0
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)

    for idx, (bar, count) in enumerate(zip(bars_total, total_class_counts)):
        pct = 100 * count / total_all if total_all else 0
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)


    plt.xticks(x, habitats, rotation=45, ha='right')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution: Train, Validation, and Total')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(destination_dir, "Total_Class_distribution.png"))
    plt.close()


def plot_sexe_distribution(sexe, destination_dir):

    all_train_sexe, all_valid_sexe = sexe

    num_classes = 2 
    x = 1
    width = 0.25  

    train_sexe_counts = np.zeros(num_classes)
    valid_sexe_counts = np.zeros(num_classes)

    for l  in all_train_sexe:
        train_sexe_counts[l] += 1
    for l in all_valid_sexe:
        valid_sexe_counts[l] += 1
    
    train_sexe_counts = train_sexe_counts[1]
    valid_sexe_counts = valid_sexe_counts[1]

    print(x)
    plt.figure(figsize=(10, 8))
    plt.bar(x - width/2, train_sexe_counts/len(all_train_sexe), width, label='Train')
    plt.bar(x + width/2, valid_sexe_counts/len(all_valid_sexe), width, label='Validation')

    
    plt.text(x-width/2, (train_sexe_counts/len(all_train_sexe)), str(train_sexe_counts)+"/"+str(len(all_train_sexe))+ f" ({train_sexe_counts/len(all_train_sexe)})", ha='center', va='bottom',fontsize=8)
    plt.text(x+width/2, (valid_sexe_counts/len(all_valid_sexe)), str(valid_sexe_counts)+"/"+str(len(all_valid_sexe))+ f" ({valid_sexe_counts/len(all_valid_sexe)})", ha='center', va='bottom',fontsize=8)

    plt.ylabel('Number of Samples')
    plt.title('Male Distribution in Train and Validation Sets')
    # plt.xticks(x, [0,1], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"Sexe_distribution"))
    plt.close()