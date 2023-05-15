import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def read_in_data(data_origin):
    """

    :param data_origin: Can be either "static" or "qoala".
    :return:
    """
    big_df = pd.DataFrame()
    for f in os.listdir("results"):
        if data_origin in f:
            df = pd.read_csv(f"results/{f}")
            big_df = pd.concat([big_df, df], ignore_index=True)
    big_df.ns_id = pd.to_numeric(big_df.ns_id, errors="coerce")
    return big_df


def plot_success_metric_per_network_schedule(dataset_id, n_sessions, ns_id, success_metric, save_fig=True):
    success_metric_str = success_metric.replace("_", " ").capitalize()

    data = read_in_data("qoala")
    filtered = data[data["dataset_id"] == dataset_id]
    filtered = filtered[filtered["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"] == ns_id]
    # TODO: do we exclude node schedules that didn't work?
    # data_filtered = data_filtered[data_filtered["makespan"] > 0]

    grouped_by_schedule_type = filtered.groupby(["schedule_type"])[success_metric]
    mean_over_schedule_type = grouped_by_schedule_type.mean().reset_index()
    std_over_schedule_type = grouped_by_schedule_type.std().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="schedule_type", y=success_metric, data=mean_over_schedule_type)
    ax.errorbar(x=mean_over_schedule_type.index, y=mean_over_schedule_type[success_metric],
                yerr=std_over_schedule_type[success_metric], fmt='none', c='black', capsize=5,
                capthick=2, elinewidth=2)
    plt.xlabel("Schedule type", fontsize=14)
    plt.ylabel(success_metric_str, fontsize=14)
    plt.title(
        f"{success_metric_str} for node schedules from dataset {dataset_id} with {n_sessions} sessions\n"
        f"based on network schedule with ID {ns_id}",
        fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/{success_metric}_dataset-{dataset_id}_sessions-{n_sessions}_ns-{ns_id}.png")


def plot_success_metric_per_dataset_ns(dataset_id, n_sessions, success_metric, save_fig=True):
    success_metric_str = success_metric.replace("_", " ").capitalize()

    data = read_in_data("qoala")
    filtered = data[data["dataset_id"] == dataset_id]
    filtered = filtered[filtered["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].notnull()]
    # TODO: do we exclude node schedules that didn't work?
    filtered = filtered[filtered["makespan"] > 0]

    complete_std = filtered.groupby(["schedule_type"])[success_metric].std().reset_index()

    grouped_by_ns = filtered.groupby(["ns_id", "schedule_type"])[success_metric]
    mean_over_ns = grouped_by_ns.mean().reset_index()

    grouped_by_schedule_type = mean_over_ns.groupby(["schedule_type"])[success_metric]
    mean_over_schedule_type = grouped_by_schedule_type.mean().reset_index()
    std_over_schedule_type = grouped_by_schedule_type.std().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="schedule_type", y=success_metric, data=mean_over_schedule_type)
    ax.errorbar(x=mean_over_schedule_type.index, y=mean_over_schedule_type[success_metric],
                yerr=complete_std[success_metric], fmt='none', c='red', capsize=6,
                capthick=3, elinewidth=3, alpha=0.4)
    ax.errorbar(x=mean_over_schedule_type.index, y=mean_over_schedule_type[success_metric],
                yerr=std_over_schedule_type[success_metric], fmt='none', c='black', capsize=5,
                capthick=2, elinewidth=2)
    plt.xlabel("Schedule type", fontsize=14)
    plt.ylabel(success_metric_str, fontsize=14)
    plt.title(
        f"{success_metric_str} for node schedules from dataset {dataset_id} with {n_sessions} sessions\n"
        f"based on randomly generated network schedules",
        fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/{success_metric}_dataset-{dataset_id}_sessions-{n_sessions}_ns-True.png")


def plot_success_metric_per_dataset_no_ns(dataset_id, n_sessions, success_metric, save_fig=True):
    success_metric_str = success_metric.replace("_", " ").capitalize()

    data = read_in_data("qoala")
    filtered = data[data["dataset_id"] == dataset_id]
    filtered = filtered[filtered["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].isnull()]
    # TODO: do we exclude node schedules that didn't work?
    # data_filtered = data_filtered[data_filtered["makespan"] > 0]

    grouped_by_schedule_type = filtered.groupby(["schedule_type"])[success_metric]
    mean_over_schedule_type = grouped_by_schedule_type.mean().reset_index()
    std_over_schedule_type = grouped_by_schedule_type.std().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="schedule_type", y=success_metric, data=mean_over_schedule_type)
    ax.errorbar(x=mean_over_schedule_type.index, y=mean_over_schedule_type[success_metric],
                yerr=std_over_schedule_type[success_metric], fmt='none', c='black', capsize=5,
                capthick=2, elinewidth=2)
    plt.xlabel("Schedule type", fontsize=14)
    plt.ylabel(success_metric_str, fontsize=14)
    plt.title(
        f"{success_metric_str} for node schedules from dataset {dataset_id} with {n_sessions} sessions\n"
        f"constructed without network schedule constraints",
        fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/{success_metric}_dataset-{dataset_id}_sessions-{n_sessions}_ns-False.png")


def plot_success_metric_all_datasets_ns(n_sessions, success_metric, save_fig=True):
    success_metric_str = success_metric.replace("_", " ").capitalize()

    data = read_in_data("qoala")
    filtered = data[data["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].notnull()]
    # TODO: do we exclude node schedules that didn't work?
    filtered = filtered[filtered["makespan"] > 0]

    complete_std = filtered.groupby(["dataset_id", "schedule_type"])[success_metric].std().reset_index()

    grouped_by_ns = filtered.groupby(["dataset_id", "ns_id", "schedule_type"])[success_metric]
    mean_over_ns = grouped_by_ns.mean().reset_index()

    grouped_by_schedule_type = mean_over_ns.groupby(["dataset_id", "schedule_type"])[success_metric]
    mean_over_schedule_type = grouped_by_schedule_type.mean().reset_index()
    std_over_schedule_type = grouped_by_schedule_type.std().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="dataset_id", y=success_metric, hue="schedule_type", data=mean_over_schedule_type)

    # Calculate the positions of the error bars
    dataset_ids = mean_over_schedule_type["dataset_id"].unique()
    n_schedule_types = len(mean_over_schedule_type["schedule_type"].unique())
    bar_width = 0.8 / n_schedule_types
    offset = np.arange(n_schedule_types) * bar_width - (bar_width * (n_schedule_types - 1) / 2)

    for i, dataset_id in enumerate(dataset_ids):
        x = np.full(n_schedule_types, i) + offset
        y = mean_over_schedule_type[mean_over_schedule_type["dataset_id"] == dataset_id][success_metric]
        yerr = complete_std[complete_std["dataset_id"] == dataset_id][success_metric]
        yerr2 = std_over_schedule_type[std_over_schedule_type["dataset_id"] == dataset_id][success_metric]
        ax.errorbar(x=x, y=y, yerr=yerr, fmt='none', c='red', capsize=6,
                capthick=3, elinewidth=3, alpha=0.4)
        ax.errorbar(x=x, y=y,
                    yerr=yerr2, fmt='none', c='black', capsize=5,
                    capthick=2, elinewidth=2)

    plt.xlabel("Dataset ID", fontsize=14)
    plt.ylabel(success_metric_str, fontsize=14)
    plt.title(
        f"{success_metric_str} for node schedules with {n_sessions} sessions\n"
        f"based on randomly generated network schedules",
        fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/{success_metric}_sessions-{n_sessions}_ns-True.png")


def plot_success_metric_all_datasets_no_ns(n_sessions, success_metric, save_fig=True):
    success_metric_str = success_metric.replace("_", " ").capitalize()

    data = read_in_data("qoala")
    filtered = data[data["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].isnull()]
    # TODO: do we exclude node schedules that didn't work?
    # filtered = filtered[filtered["makespan"] > 0]

    grouped_by_schedule_type = filtered.groupby(["dataset_id", "schedule_type"])[success_metric]
    mean_over_schedule_type = grouped_by_schedule_type.mean().reset_index()
    std_over_schedule_type = grouped_by_schedule_type.std().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="dataset_id", y=success_metric, hue="schedule_type", data=mean_over_schedule_type)

    # Calculate the positions of the error bars
    dataset_ids = filtered["dataset_id"].unique()
    n_schedule_types = len(filtered["schedule_type"].unique())
    bar_width = 0.8 / n_schedule_types
    offset = np.arange(n_schedule_types) * bar_width - (bar_width * (n_schedule_types - 1) / 2)

    for i, dataset_id in enumerate(dataset_ids):
        x = np.full(n_schedule_types, i) + offset
        y = mean_over_schedule_type[mean_over_schedule_type["dataset_id"] == dataset_id][success_metric]
        yerr = std_over_schedule_type[std_over_schedule_type["dataset_id"] == dataset_id][success_metric]
        ax.errorbar(x=x, y=y, yerr=yerr, fmt='none', c='black', capsize=5,
                    capthick=2, elinewidth=2)

    plt.xlabel("Dataset ID", fontsize=14)
    plt.ylabel(success_metric_str, fontsize=14)
    plt.title(
        f"{success_metric_str} for node schedules with {n_sessions} sessions\n"
        f"constructed without network schedule constraints",
        fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

    if save_fig:
        plt.savefig(f"plots/{success_metric}_sessions-{n_sessions}_ns-False.png")


def plot_makespan_comparison(n_sessions, schedule_type, ns=True):
    qoala = read_in_data("qoala")
    static = read_in_data("static")

    qoala_filtered = qoala[qoala.n_sessions == n_sessions]
    static_filtered = static[static.n_sessions == n_sessions]

    qoala_filtered = qoala_filtered[qoala_filtered.schedule_type == schedule_type]
    static_filtered = static_filtered[static_filtered.schedule_type == schedule_type]

    qoala_filtered = qoala_filtered[qoala_filtered.makespan > 0]
    static_filtered = static_filtered[static_filtered.makespan > 0]

    if ns:
        qoala_filtered = qoala_filtered[qoala_filtered.ns_id.notnull()]
        static_filtered = static_filtered[static_filtered.ns_id.notnull()]
    else:
        qoala_filtered = qoala_filtered[qoala_filtered.ns_id.isnull()]
        static_filtered = static_filtered[static_filtered.ns_id.isnull()]

    qoala_grouped = qoala_filtered.groupby(["dataset_id"])["makespan"]
    qoala_mean = qoala_grouped.mean().reset_index()
    qoala_mean["origin"] = ["dynamic"] * len(qoala_mean.makespan)
    qoala_std = qoala_grouped.std().reset_index()

    static_grouped = static_filtered.groupby(["dataset_id"])["makespan"]
    static_mean = static_grouped.mean().reset_index()
    static_mean["origin"] = ["static"] * len(static_mean.makespan)
    static_std = static_grouped.mean().reset_index()

    mean = pd.concat([static_mean, qoala_mean], ignore_index=True)
    std = pd.concat([static_std, qoala_std], ignore_index=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="dataset_id", y="makespan", data=mean, hue="origin")

    # Calculate the positions of the error bars
    dataset_ids = mean["dataset_id"].unique()
    n_schedule_types = len(mean["origin"].unique())
    bar_width = 0.8 / n_schedule_types
    offset = np.arange(n_schedule_types) * bar_width - (bar_width * (n_schedule_types - 1) / 2)

    for i, dataset_id in enumerate(dataset_ids):
        x = np.full(n_schedule_types, i) + offset
        y = mean[mean["dataset_id"] == dataset_id]["makespan"]
        yerr = std[std["dataset_id"] == dataset_id]["makespan"]
        ax.errorbar(x, y, yerr=yerr, fmt='none', c='black', capsize=4, alpha=0.7)

    plt.xlabel("Dataset ID", fontsize=14)
    plt.ylabel("Makespan", fontsize=14)
    plt.title(
        f"Makespan of {schedule_type} node schedules from all datasets with {n_sessions} sessions\n"
        f"{'using randomly generated network schedules' if ns else 'without network schedule constraints'}",
        fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()


if __name__ == '__main__':
    # plot_makespan_comparison(6, "NAIVE", ns=False)
    # plot_success_metric_per_network_schedule(0, 6, 8, "success_probability")
    # plot_success_metric_per_dataset_ns(0, 6, "success_probability")
    # plot_success_metric_per_dataset_no_ns(0, 6, "success_probability")
    # plot_success_metric_all_datasets_ns(6, "makespan")
    # plot_success_metric_all_datasets_no_ns(6, "success_probability", save_fig=False)
    plot_makespan_comparison(6, "NAIVE", ns=False)
