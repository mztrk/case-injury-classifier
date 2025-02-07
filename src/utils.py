import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_confusion_matrix(conf_matrix, filename, title="Confusion Matrix"):
    """
    Saves the confusion matrix as a PNG file in the ./docs/ folder.

    Parameters:
        conf_matrix (numpy.ndarray): The confusion matrix.
        filename (str): The name of the file to save (without extension).
        title (str): The title of the plot.
    """
    # Create docs directory if it doesn't exist
    os.makedirs("./docs", exist_ok=True)

    # Define file path
    file_path = f"./docs/{filename}.png"

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Injury", "Injury"], yticklabels=["No Injury", "Injury"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    # Save the figure
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Confusion matrix saved at: {file_path}")

