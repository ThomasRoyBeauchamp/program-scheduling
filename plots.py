import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def read_in_data(data_origin):
    big_df = pd.DataFrame()
    for f in os.listdir("results"):
        if data_origin in f:
            df = pd.read_csv(f"results/{f}")
            big_df = pd.concat([big_df, df], ignore_index=True)
    big_df.ns_id = pd.to_numeric(big_df.ns_id, errors="coerce")
    return big_df

def set_up_seaborn():
    sns.set_theme(context={'axes.linewidth': 1,
                           'grid.linewidth': 1,
                           'lines.linewidth': 1,
                           'lines.markersize': 1,
                           'patch.linewidth': 1,
                           'xtick.major.width': 1,
                           'ytick.major.width': 1,
                           'xtick.minor.width': 0.8,
                           'ytick.minor.width': 0.8,
                           'xtick.major.size': 5,
                           'ytick.major.size': 3,
                           'xtick.minor.size': 5,
                           'ytick.minor.size': 3,
                           'font.size': 20,
                           'axes.labelsize': 20,
                           'axes.titlesize': 20,
                           'xtick.labelsize': 18,
                           'ytick.labelsize': 18,
                           'legend.fontsize': 20,
                           'legend.title_fontsize': 20},
                  style="ticks",
                  palette=['#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3'],
                  font_scale=2,
                  rc={"text.usetex": True})


def plot_heuristic_makespan_for_one_network_schedule(n_sessions, schedule_type, save_fig=True):
    qoala = read_in_data("qoala")
    static = read_in_data("static")

    data = {
        "dataset_id": [],
        "makespan": [],
        "ns_id": [],
        "origin": [],
        "std": []
    }

    static_filtered = static[static.n_sessions == n_sessions]
    static_filtered = static_filtered[static_filtered.schedule_type == schedule_type]

    qoala_filtered = qoala[qoala.n_sessions == n_sessions]
    qoala_filtered = qoala_filtered[qoala_filtered.schedule_type == schedule_type]

    if n_sessions == 6:
        ns_ids = [8, 353, 0, 17, 23, 23, 58]
    elif n_sessions == 12:
        ns_ids = [25, 89, 0, 180, 170, 170, 208]

    for id, ns in zip(list(range(7)), ns_ids):
        new_static_filtered = static_filtered[static_filtered.dataset_id == id]
        new_static_filtered = new_static_filtered[new_static_filtered.ns_id == ns]

        new_qoala_filtered = qoala_filtered[qoala_filtered.dataset_id == id]
        new_qoala_filtered = new_qoala_filtered[new_qoala_filtered.ns_id == ns]

        static_grouped = new_static_filtered.groupby(["role"])["makespan"]
        static_mean = static_grouped.mean().reset_index()

        qoala_mean = new_qoala_filtered["makespan"].mean()
        qoala_std = new_qoala_filtered["makespan"].std()

        data["dataset_id"] += [id, id, id]
        data["ns_id"] += [ns, ns, ns]
        data["origin"] += ["Static (Alice)", "Static (Bob)", "Dynamic"]
        data["makespan"] += [int(static_mean[static_mean.role == "alice"]["makespan"].values[0])/1e9,
                             int(static_mean[static_mean.role == "bob"]["makespan"].values[0])/1e9,
                             qoala_mean/1e9]
        data["std"] += [0, 0, qoala_std/1e9]

    plt.figure(figsize=(12, 6))

    if n_sessions == 6:
        plt.ylim((0.8, 1.8))
    elif n_sessions == 12:
        plt.ylim((3.5, 5.5))

    ax = sns.barplot(x="dataset_id", y="makespan", data=data, hue="origin")

    # Calculate the positions of the error bars
    dataset_ids = list(range(7))
    n_schedule_types = 3
    bar_width = 0.8 / n_schedule_types
    offset = np.arange(n_schedule_types) * bar_width - (bar_width * (n_schedule_types - 1) / 2)

    for i, dataset_id in enumerate(dataset_ids):
        x = np.full(n_schedule_types, i) + offset
        y = [data["makespan"][i] for i in range(len(data["makespan"])) if data["dataset_id"][i] == dataset_id]
        yerr = [data["std"][i] for i in range(len(data["std"])) if data["dataset_id"][i] == dataset_id]
        for x1, y1, yerr1 in zip(x, y, yerr):
            if yerr1 > 0:
                ax.errorbar(x1, y1, yerr=yerr1, fmt='none', c='black', capsize=4, alpha=0.7)

    ax.set_xticklabels(["BQC", "PP", "QKD", "BQC\n\& PP", "BQC\n\& QKD", "PP\n\& QKD", "BQC \& PP\n\& QKD"])
    plt.xlabel("Dataset")
    plt.ylabel("Makespan (s)")
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"plots/compare_makespan_{schedule_type}_{n_sessions}_from_one_NS.png")
    else:
        plt.show()


def plot_success_probability_per_network_schedule(dataset_id, n_sessions, ns_id, success_metric, save_fig=True):
    data = read_in_data("qoala")
    filtered = data[data["dataset_id"] == dataset_id]
    filtered = filtered[filtered["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"] == ns_id]

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=filtered, x="schedule_type", y="success_probability", errorbar="sd",
                     capsize=0.1, errwidth=1, errcolor="black")
    plt.xlabel("Schedule type")
    plt.ylabel("Average success probability")
    plt.tight_layout()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/{success_metric}_dataset-{dataset_id}_sessions-{n_sessions}_ns-{ns_id}.png")
    else:
        plt.show()


def plot_success_probability_per_dataset_ns(dataset_id, n_sessions, save_fig=True):
    data = read_in_data("qoala")
    filtered = data[data["dataset_id"] == dataset_id]
    filtered = filtered[filtered["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].notnull()]

    success_metric = "success_probability"
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
                yerr=complete_std[success_metric], capsize=16, elinewidth=2, c='red',
                alpha=0.4, fmt='none', capthick=2)
    ax.errorbar(x=mean_over_schedule_type.index, y=mean_over_schedule_type[success_metric],
                yerr=std_over_schedule_type[success_metric], fmt='none', c='black', capsize=8,
                capthick=1, elinewidth=2)

    plt.xlabel("Schedule type")
    plt.ylabel("Average success probability")
    plt.tight_layout()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/success-probability_dataset-{dataset_id}_sessions-{n_sessions}_ns-True.png")
    else:
        plt.show()


def plot_success_probability_all_datasets_ns(n_sessions, save_fig=True):
    success_metric = "success_probability"
    data = read_in_data("qoala")
    filtered = data[data["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].notnull()]
    filtered = filtered[filtered["schedule_type"].isin(["HEU", "OPT"])]

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
        ax.errorbar(x=x, y=y, yerr=yerr, fmt='none', c='red', capsize=8,
                    capthick=2, elinewidth=2, alpha=0.4)
        ax.errorbar(x=x, y=y,
                    yerr=yerr2, fmt='none', c='black', capsize=6,
                    capthick=2, elinewidth=2)

    ax.set_xticklabels(["BQC", "PP", "QKD", "BQC\n\& PP", "BQC\n\& QKD", "PP\n\& QKD", "BQC \& PP\n\& QKD"])
    plt.xlabel("Dataset")
    plt.ylabel("Average success probability")
    plt.legend(title="Schedule type")
    plt.tight_layout()

    # Save the bar graph as an image
    if save_fig:
        plt.savefig(f"plots/{success_metric}_sessions-{n_sessions}_ns-True.png")
    else:
        plt.show()


def compare_solve_times_for_ns(save_fig=True):
    static = read_in_data("static")

    static_filtered = static[static.schedule_type != "NAIVE"]
    static_filtered = static_filtered[static_filtered.ns_id.notnull()]
    static_filtered["temp"] = static_filtered["n_sessions"].astype(str) + " sessions, " + static_filtered["schedule_type"]

    plt.figure(figsize=(12, 6))

    ax = sns.barplot(x="dataset_id", y="solve_time", data=static_filtered, hue="temp",
                     capsize=0.1, errwidth=1, errcolor="black", errorbar="sd")

    plt.legend(title=None)
    ax.set_xticklabels(["BQC", "PP", "QKD", "BQC\n\& PP", "BQC\n\& QKD", "PP\n\& QKD", "BQC \& PP\n\& QKD"])
    plt.xlabel("Dataset")
    plt.ylabel("Time to construct a node schedule (s)")
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"plots/compare_solve_time_ns.png")
    else:
        plt.show()


def plot_if_for_solve_time_for_ns(save_fig=True):
    static = read_in_data("static")

    static_filtered = static[static.schedule_type != "NAIVE"]
    static_filtered = static_filtered[static_filtered.ns_id.notnull()]

    grouped = static_filtered.groupby(["dataset_id", "schedule_type", "n_sessions"])["solve_time"]

    mean = grouped.mean().reset_index()
    std = grouped.std().reset_index()

    data = {
        "dataset_id": [],
        "n_sessions": [],
        "if": [],
        "min": [],
        "max": []
    }

    for d in mean.dataset_id.unique():
        for n in mean.n_sessions.unique():
            heu = float(mean[mean.dataset_id == d][mean.n_sessions == n][mean.schedule_type == "HEU"]["solve_time"])
            opt = float(mean[mean.dataset_id == d][mean.n_sessions == n][mean.schedule_type == "OPT"]["solve_time"])
            heu_std = float(std[std.dataset_id == d][std.n_sessions == n][std.schedule_type == "HEU"]["solve_time"])
            opt_std = float(std[std.dataset_id == d][std.n_sessions == n][std.schedule_type == "OPT"]["solve_time"])

            data["dataset_id"].append(d)
            data["n_sessions"].append(n)
            if_value = heu/opt
            data["if"].append(if_value)
            data["max"].append(((heu + heu_std) / (opt - opt_std)) - if_value)
            data["min"].append(if_value - ((heu - heu_std) / (opt + opt_std)))

    plt.figure(figsize=(12, 6))

    ax = sns.pointplot(x="dataset_id", y="if", data=data, hue="n_sessions", markers=["o", "D"], dodge=True, join=False)

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    x_coords = [x_coord for x_coord in x_coords if not np.ma.is_masked(x_coord)]
    y_coords = [y_coord for y_coord in y_coords if not np.ma.is_masked(y_coord)]

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over
    colors = ['#8da0cb'] * 7 + ['#fc8d62'] * 7
    for x, y, dmin, dmax, color in zip(x_coords, y_coords, data["min"], data["max"], colors):
        ax.errorbar(x, y, yerr=[[dmin], [dmax]],
                    ecolor=color, fmt=' ', zorder=-1, label='_nolegend_', capsize=3)

    ax.set_xticklabels(["BQC", "PP", "QKD", "BQC\n\& PP", "BQC\n\& QKD", "PP\n\& QKD", "BQC \& PP\n\& QKD"])
    plt.xlabel("Dataset")
    plt.ylabel("Improvement factor of solve time\n(HEU / OPT)")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=["6 sessions", "12 sessions"])
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"plots/if_solve_time_ns.png")
    else:
        plt.show()


def plot_success_probability_all_datasets_no_ns(n_sessions, save_fig=True):
    success_metric = "success_probability"

    data = read_in_data("qoala")
    filtered = data[data["n_sessions"] == n_sessions]
    filtered = filtered[filtered["ns_id"].isnull()]

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=filtered, x="dataset_id", y="success_probability", hue="schedule_type", errorbar="sd",
                     capsize=0.1, errwidth=1, errcolor="black")

    ax.set_xticklabels(["BQC", "PP", "QKD", "BQC\n\& PP", "BQC\n\& QKD", "PP\n\& QKD", "BQC \& PP\n\& QKD"])
    plt.xlabel("Dataset")
    plt.ylabel("Average success probability")
    plt.legend(title="Schedule type")
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"plots/{success_metric}_sessions-{n_sessions}_ns-False.png")
    else:
        plt.show()


def plot_compare_success_metric_for_ns_and_no_ns(n_sessions, success_metric, save_fig=True):
    data = read_in_data("qoala")

    data = data[data.schedule_type == "HEU"]
    filtered = data[data.n_sessions == n_sessions]
    if success_metric == "makespan":
        filtered["makespan"] = filtered["makespan"]/1e9
    filtered.loc[filtered['ns_id'].isnull(), 'has_ns'] = False
    filtered.loc[filtered['ns_id'].notnull(), 'has_ns'] = True

    means = filtered.groupby(["dataset_id", "has_ns"])[success_metric].mean().reset_index()
    std = filtered.groupby(["dataset_id", "has_ns"])[success_metric].std().reset_index()

    # Create a bar graph using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="dataset_id", y=success_metric, hue="has_ns", data=means)

    # Calculate the positions of the error bars
    dataset_ids = means["dataset_id"].unique()
    n_schedule_types = 2
    bar_width = 0.8 / n_schedule_types
    offset = np.arange(n_schedule_types) * bar_width - (bar_width * (n_schedule_types - 1) / 2)

    for i, dataset_id in enumerate(dataset_ids):
        x = np.full(n_schedule_types, i) + offset
        y = means[means["dataset_id"] == dataset_id][success_metric]
        yerr = std[std["dataset_id"] == dataset_id][success_metric]
        for x1, y1, yerr1 in zip(x, y, yerr):
            if yerr1 > 0:
                ax.errorbar(x=x1, y=y1, yerr=yerr1, fmt='none', c='black', capsize=8,
                            capthick=2, elinewidth=2, alpha=0.4)

    ax.set_xticklabels(["BQC", "PP", "QKD", "BQC\n\& PP", "BQC\n\& QKD", "PP\n\& QKD", "BQC \& PP\n\& QKD"])
    plt.xlabel("Dataset")
    handles, labels = ax.get_legend_handles_labels()
    if success_metric == "makespan":
        plt.ylabel("Makespan (s)")
        plt.legend(handles=handles, title=None, labels=["No NS", "With NS"], loc="upper left")
    elif success_metric == "success_probability":
        plt.xlabel("Average success probability")
        plt.legend(handles=handles, title=None, labels=["No NS", "With NS"], loc="lower left")
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"plots/ns_comparison-HEU-{success_metric}.png")
    else:
        plt.show()


if __name__ == '__main__':
    set_up_seaborn()

    # Plots for -- 7.1 Effect of Asynchronous Classical Communication --
    plot_heuristic_makespan_for_one_network_schedule(6, "HEU")
    plot_heuristic_makespan_for_one_network_schedule(12, "HEU")

    # Plots for -- 7.2 Heuristic-Driven and Optimal Node Schedules --
    plot_success_probability_per_network_schedule(0, 6, 8, "success_probability")
    plot_success_probability_per_dataset_ns(dataset_id=0, n_sessions=6)
    plot_success_probability_all_datasets_ns(6)
    plot_success_probability_all_datasets_ns(12)
    compare_solve_times_for_ns()
    plot_if_for_solve_time_for_ns()

    # Plots for -- 7.3 Node Schedules Without Network Schedule Constraints --
    plot_success_probability_all_datasets_no_ns(n_sessions=12)
    plot_compare_success_metric_for_ns_and_no_ns(12, "success_probability")
    plot_compare_success_metric_for_ns_and_no_ns(12, "makespan")
