import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_with_target(
    file_path, target_column="default", output_image="correlation_heatmap.png"
):
    # Load dataset
    df = pd.read_csv(file_path)

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Extract the correlation with the target column
    corr_with_target = corr_matrix[[target_column]].sort_values(
        by=target_column, ascending=False
    )

    # Plot heatmap for correlation with target
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_with_target, annot=True, cmap="coolwarm", cbar=True, vmin=-1, vmax=1
    )

    # Title and labels
    plt.title(f"Correlation with {target_column}", fontsize=16)
    plt.tight_layout()

    # Save the image
    plt.savefig(output_image)
    print(f"Correlation heatmap saved as {output_image}")


if __name__ == "__main__":
    # File path to your dataset
    file_path = "Loan_Data.csv"

    # Call the function to generate the correlation heatmap
    correlation_with_target(file_path)
