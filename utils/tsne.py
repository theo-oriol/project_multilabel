import numpy as np 
import colorsys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os 

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import colorsys

def tsne(features, labels, destination_dir):
    def generate_two_colors():
        # color for class-positive and class-negative
        return [
            colorsys.hsv_to_rgb(0.6, 0.65, 0.9),  # positive (highlight)
            colorsys.hsv_to_rgb(0.0, 0.0, 0.7)    # negative (dimmed gray)
        ]

    # Create subdirectory to save plots
    tsne_dir = os.path.join(destination_dir, "tsne")
    os.makedirs(tsne_dir, exist_ok=True)

    # Compute t-SNE projection
    tsne_result = TSNE(
        n_components=2, 
        learning_rate='auto', 
        init='random', 
        perplexity=10
    ).fit_transform(features.squeeze())

    num_classes = labels.shape[1]

    for class_idx in range(num_classes):
        class_labels = labels[:, class_idx]  # binary vector for this class

        plt.figure(figsize=(8, 6))
        idx_pos = class_labels == 1
        idx_neg = class_labels == 0

        colors = generate_two_colors()

        # Plot negative class (gray)
        plt.scatter(tsne_result[idx_neg, 0], tsne_result[idx_neg, 1],
                    c=[colors[1]], label="Not Present", alpha=0.4, edgecolors='none')
        # Plot positive class (colored)
        plt.scatter(tsne_result[idx_pos, 0], tsne_result[idx_pos, 1],
                    c=[colors[0]], label="Present", alpha=0.8, edgecolors='k')

        plt.title(f"t-SNE: Class {class_idx}")
        plt.legend(title="Class Presence", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(tsne_dir, f"tsne_class_{class_idx}.png")
        plt.savefig(save_path)
        plt.close()
