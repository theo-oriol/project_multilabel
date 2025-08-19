import numpy as np 
import os 
import matplotlib.pyplot as plt
import colorsys
import random
import math
from sklearn.metrics import precision_score, recall_score
import torch 
from torcheval.metrics import MultilabelPrecisionRecallCurve

def generate_n_colors(n):
        hues = np.linspace(0, 1, n, endpoint=False)
        saturations = np.linspace(0.6, 0.9, n)
        values = np.linspace(0.8, 1.0, n)

        colors = [
            colorsys.hsv_to_rgb(h, s, v)
            for h, s, v in zip(hues, saturations, values)
        ]
        return colors

def plot_loss(epochs,metrics,destination_dir):
    (train_metrics, valid_metrics) = metrics


    train_macro_species_acc = []
    for i in range(len(train_metrics.metrics["species_macro_accuracy"])):
        train_macro_species_acc.append(np.mean(list(train_metrics.metrics["species_macro_accuracy"][i].values())))

    valid_macro_species_acc = []
    for i in range(len(valid_metrics.metrics["species_macro_accuracy"])):
        valid_macro_species_acc.append(np.mean(list(valid_metrics.metrics["species_macro_accuracy"][i].values())))

    train_acc_female = []
    train_acc_male = []
    for i in range(len(train_metrics.metrics["sexe_macro_accuracy"])):
        train_acc_female.append(train_metrics.metrics["sexe_macro_accuracy"][i][0])
        train_acc_male.append(train_metrics.metrics["sexe_macro_accuracy"][i][1])


    valid_acc_female = []
    valid_acc_male = []
    for i in range(len(valid_metrics.metrics["sexe_macro_accuracy"])):
        valid_acc_female.append(valid_metrics.metrics["sexe_macro_accuracy"][i][0])
        valid_acc_male.append(valid_metrics.metrics["sexe_macro_accuracy"][i][1])

    e = [i for i in range(epochs)]
    
    plt.figure(figsize=(18, 10))

    plt.subplot(3, 2, 1)
    plt.plot(e, train_metrics.metrics["loss"], label="Train Loss")
    plt.plot(e, valid_metrics.metrics["loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(e, train_metrics.metrics["accuracy"], label="Train Accuracy")
    plt.plot(e, valid_metrics.metrics["accuracy"], label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(e, train_metrics.metrics["macro_accuracy"], label="Train Macro Accuracy")
    plt.plot(e, valid_metrics.metrics["macro_accuracy"], label="Valid Macro Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Macro Accuracy")
    plt.title("Macro Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(e, train_macro_species_acc, label="Train Macro SPE Accuracy")
    plt.plot(e, valid_macro_species_acc, label="Valid Macro SPE Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Macro Species Accuracy")
    plt.title("Macro Species Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(e, train_acc_female, label="Train female Accuracy")
    plt.plot(e, train_acc_male, label="Train male Accuracy")
    plt.plot(e, valid_acc_female, label="Valid female Accuracy")
    plt.plot(e, valid_acc_male, label="Valid male Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Sexe Accuracy")
    plt.title("Sexe Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"losses"))






def pression_recall(all_valid_real_prob,all_valid_labels,habitat,destination_dir):
    metric = MultilabelPrecisionRecallCurve(num_labels=all_valid_real_prob.shape[1])
    metric.update(torch.from_numpy(all_valid_real_prob), torch.from_numpy(all_valid_labels))
    precision, recall, _ = metric.compute()

    
    colors = generate_n_colors(len(precision))
    random.shuffle(colors)

    recall_grid = np.linspace(0, 1, 500)
    interp_precisions = []
    ap_per_class = []
    plt.figure(figsize=(10,5))

    for i in range(len(precision)):
        # Sort by recall
        rec = np.array(recall[i])
        prec = np.array(precision[i])
        sorted_idx = np.argsort(rec)
        rec = rec[sorted_idx]
        prec = prec[sorted_idx]
        
        # Interpolate onto common grid (fill missing values with 0)
        interp = np.interp(recall_grid, rec, prec, left=0, right=0)
        interp_precisions.append(interp)
        
        plt.plot(rec, prec, label=f"{habitat[i]} {np.mean(prec):.3f}", color=colors[i])
        ap = np.trapz(prec, rec)
        ap_per_class.append(ap)

    interp_precisions = np.array(interp_precisions)
    mAP_curve = np.mean(interp_precisions, axis=0)
    mAP_value = np.mean(mAP_curve)

    plt.plot(recall_grid, mAP_curve, linewidth=4, color='black', label=f"mAP {mAP_value:.3f}")

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.ylim(0, 1)
    plt.legend(title="Class",loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(destination_dir,"PR"))


    x = np.arange(len(habitat))
    ap_per_class = np.array(ap_per_class)
    images_per_class = np.sum(all_valid_labels, axis=0)
    images_per_class = np.array(images_per_class)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(x, ap_per_class, color='steelblue', label='AP per Class')
    ax1.set_ylabel("Average Precision (AP)")
    ax1.set_xlabel("Habitat")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(habitat, rotation=45, ha='right')
    ax1.grid(True, axis='y')

    # Secondary y-axis for image count
    ax2 = ax1.twinx()
    ax2.plot(x, images_per_class, 'r--o', label='Image Count', linewidth=2)
    ax2.set_ylabel("Number of Images", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir, "AP_per_class_with_image_count.png"))
    plt.close()


def families_plot(all_valid_real_prob,all_valid_labels,all_valid_family,habitat,families,destination_dir):

    map_per_env_per_family = []
    img_per_env_per_family = []
    
    for i in np.unique(all_valid_family):
        mask = all_valid_family == i
        
        y_true = torch.from_numpy(all_valid_labels[mask])
        y_pred = torch.from_numpy(all_valid_real_prob[mask])

        img_per_env_per_family.append(len(all_valid_labels[mask]))

        metric = MultilabelPrecisionRecallCurve(num_labels=all_valid_real_prob.shape[1])
        metric.update(y_pred, y_true)
        precision, recall, _ = metric.compute()

        

        map_per_env = []
        for i in range(len(precision)):
        
            assert recall[i].shape == precision[i].shape, "recall and precision must be the same shape"

            rec = recall[i]
            prec = precision[i]
            sorted_idx = np.argsort(rec)
            rec = rec[sorted_idx]
            prec = prec[sorted_idx]
            ap = np.trapz(prec, rec)
            map_per_env.append(ap)

        map_per_env_per_family.append(np.mean(map_per_env))
        


    map_per_env_per_family = np.array(map_per_env_per_family)  
    img_per_env_per_family = np.array(img_per_env_per_family)


    plt.figure()
    plt.plot(img_per_env_per_family, map_per_env_per_family, 'bo', markersize=3)
    plt.xlabel("Count")
    plt.ylabel("AP (per family)")
    plt.grid(True)

    plt.title("Distribution of mAP per Family", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(os.path.join(destination_dir,"Distribution of mAP per Family"))


def species_plot(all_valid_real_prob,all_valid_labels,all_valid_spe,destination_dir):
    map_per_env_per_species = []
    img_per_env_per_species = []

    for i in np.unique(all_valid_spe):
        mask = all_valid_spe == i

        y_true = torch.from_numpy(all_valid_labels[mask])
        y_pred = torch.from_numpy(all_valid_real_prob[mask])
        img_per_env_per_species.append(len(all_valid_labels[mask]))

        metric = MultilabelPrecisionRecallCurve(num_labels=all_valid_labels[mask].shape[1])
        metric.update(y_pred, y_true)
        precision, recall, _ = metric.compute()

        precision = np.array(precision)
        recall = np.array(recall)

        ap_per_env = []

        for i in range(len(precision)):
        
            assert recall[i].shape == precision[i].shape, "recall and precision must be the same shape"

            rec = recall[i]
            prec = precision[i]
            sorted_idx = np.argsort(rec)
            rec = rec[sorted_idx]
            prec = prec[sorted_idx]
            ap = np.trapz(prec, rec)
            ap_per_env.append(ap)

        map_per_env_per_species.append(np.mean(ap_per_env))
        

    map_per_env_per_species = np.array(map_per_env_per_species)  
    img_per_env_per_species = np.array(img_per_env_per_species)


    # plt.figure()
    # plt.hist(map_per_env_per_species, bins=20, color='skyblue', edgecolor='black')
    # plt.xlabel("Count")
    # plt.ylabel("mAP (per species)")
    # plt.grid(True)

    # plt.title("Distribution of mAP per species", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  
    # plt.savefig(os.path.join(destination_dir,"Distribution of mAP per species"))

    bins = 20
    counts, bin_edges = np.histogram(map_per_env_per_species, bins=bins)

    # Compute total images per bin
    img_counts = np.zeros_like(counts, dtype=float)
    for i in range(len(map_per_env_per_species)):
        map_val = map_per_env_per_species[i]
        img_count = img_per_env_per_species[i]
        bin_idx = np.searchsorted(bin_edges, map_val, side='right') - 1
        if 0 <= bin_idx < bins:
            img_counts[bin_idx] += img_count

    # Filter out bins with 0 images
    valid_bins = img_counts > 0
    filtered_counts = counts[valid_bins]
    filtered_img_props = img_counts[valid_bins] / img_per_env_per_species.sum()
    filtered_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    filtered_bin_centers = filtered_bin_centers[valid_bins]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_bin_centers, filtered_counts, width=(bin_edges[1] - bin_edges[0]), color='skyblue', edgecolor='black', label='Number of Species')

    # Plot line of image proportions (scaled)
    plt.plot(filtered_bin_centers, filtered_img_props * max(filtered_counts), 'r--o', label='Proportion of Images (scaled)', linewidth=2)

    # Labels and style
    plt.xlabel("mAP (per species)")
    plt.ylabel("Number of Species")
    plt.title("Histogram of mAP per Species with Image Proportion Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(destination_dir, exist_ok=True)
    plt.savefig(os.path.join(destination_dir, "Histogram_mAP_per_species.png"))
    plt.close()


def prob_distribution(all_valid_real_prob,all_valid_labels,habitats,destination_dir, ncols=5,n_bins=100):
    labels = np.asarray(all_valid_labels)
    probs = np.asarray(all_valid_real_prob)
    assert probs.shape == labels.shape, "probs and labels must have the same shape (N, C)"
    N, n = probs.shape

    colors = generate_n_colors(n)
    random.shuffle(colors)

    ncols = min(max(1, ncols), max(1, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.4), squeeze=False)

    # Shared binning across classes for consistency
    bin_edges = np.linspace(0.0, 1.0, n_bins, endpoint=True)
    bin_centers = bin_edges + 0.05

    for i in range(n):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        p = probs[:, i]
        y = labels[:, i].astype(int)

        # Histogram (counts) on left axis
        ax.hist(p, bins=bin_edges, alpha=0.45, color=colors[i], label=f"{habitats[i]}: counts")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Count")
        ax.set_title(f"{habitats[i]}")
        ax.grid(True, axis="both", alpha=0.25)

        # Precision per bin on right axis
        ax2 = ax.twinx()
        # Compute per-bin precision = TP / (TP + FP)
        precision_vals = []
        recall_vals = []
        for b in range(n_bins):
            mask = np.array((p >= bin_edges[b])).astype(int)
            precision_vals.append(precision_score(y, mask,zero_division=0))
            recall_vals.append(recall_score(y, mask,zero_division=0))

        ax2.plot(bin_centers, precision_vals, marker='-', linewidth=1.5, label="Precision per bin")
        ax2.plot(bin_centers, recall_vals, marker='-', linewidth=1.5, color="green", label="Precision per bin")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Precision")

        # Build a combined legend (hist on ax, line on ax2)
        # We fetch handles/labels from both axes
        # h1, l1 = ax.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        # if h1 or h2:
        #     ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, frameon=True)

    # Turn off any unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')

    fig.tight_layout()
    out_path = os.path.join(destination_dir, "probability_distribution_with_precision.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path