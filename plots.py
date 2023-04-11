import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_success_metric_per_dataset(param, dataset_id, save_fig=True, csv_file="node_schedule_results.csv"):
    df = pd.read_csv(csv_file)

    # Filter rows with 'name' starting with "dataset_0"
    df_filtered = df[df["name"].str.startswith("dataset_" + str(dataset_id))]

    # Group by 'role' and 'schedule_type', and then calculate the mean of 'makespan'
    df_grouped = df_filtered.groupby(["node", "schedule_type"])[param]

    for e in df_grouped.groups:
        if len(df_grouped.groups.get(e)) > 1:
            print(f"There are more results for {e[0]} of {e[1]} schedule for dataset {dataset_id}, "
                  f"the average is taken.")

    df_grouped = df_grouped.mean().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x="node", y=param, hue="schedule_type", data=df_grouped)
    plt.xlabel("Node")
    plt.ylabel(param)
    plt.title(f"{param} for Node Schedules from Dataset {dataset_id}")

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/bar_graph_{param}_dataset_{dataset_id}.png")

    # Show the bar graph
    # plt.show()


def plot_improvement_factors(param, save_fig=True, csv_file="node_schedule_results.csv"):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract dataset identifier from 'name' and create a new column 'dataset_id'
    df['dataset_id'] = df['name'].str.extract(r'(dataset_\d)')

    # Filter rows with 'name' starting with "dataset_" followed by a number between 0 and 6
    df_filtered = df[df['dataset_id'].notna()]

    # Separate rows with 'NAIVE' and 'HEU' schedule types
    df_naive = df_filtered[df_filtered["schedule_type"] == "NAIVE"]
    df_heu = df_filtered[df_filtered["schedule_type"] == "HEU"]

    # Merge the two dataframes on ['dataset_id', 'node']
    merged_df = pd.merge(df_naive, df_heu, on=['dataset_id', 'node'], suffixes=('_naive', '_heu'))

    # Calculate the improvement factor
    merged_df['improvement_factor'] = merged_df[f'{param}_heu'] / merged_df[f'{param}_naive']

    # Create a scatter plot grouped by dataset
    plt.figure(figsize=(12, 6))

    datasets = sorted(merged_df['dataset_id'].unique())
    x = np.arange(len(datasets))
    width = 0.35

    for i, dataset in enumerate(datasets):
        dataset_df = merged_df[merged_df['dataset_id'] == dataset]
        alice_improvement = dataset_df.loc[dataset_df['node'] == 'alice', 'improvement_factor'].values[0]
        bob_improvement = dataset_df.loc[dataset_df['node'] == 'bob', 'improvement_factor'].values[0]
        plt.scatter(i - width / 2, alice_improvement, color='orange', marker='o')
        plt.scatter(i + width / 2, bob_improvement, color='blue', marker='s')

    plt.xticks(x, datasets)
    plt.xlabel("Dataset")
    plt.ylabel(f"{param} Improvement Factor (HEU / NAIVE)")

    # Create the legend
    plt.legend(['Alice', 'Bob'])

    # Save the scatter plot as an image
    if save_fig:
        plt.savefig(f"plots/{param}_improvement_factors.png", bbox_inches='tight')

    # Show the scatter plot
    # plt.show()


if __name__ == '__main__':
    # for i in range(7):
    #     plot_success_metric_per_dataset("makespan", i)
    #     plot_success_metric_per_dataset("PUF_QPU", i)
    for p in ["makespan", "PUF_QPU"]:
        plot_improvement_factors(p)
