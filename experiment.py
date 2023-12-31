import pandas as pd
import numpy as np
import os
from emteqai.oco.load_data import load_data
from emteqai.utils.processing.signals.filters import bandpass_filter
import scipy
import matplotlib.pyplot as plt
from emteqai.utils.processing.data import segmentation
import peakutils
import warnings
from scipy.signal import find_peaks as scipy_find_peaks


color_dict = {"v": "tab:blue", "p": "tab:red"}


def fix_annotations(data, label_columns=None):
    # Remove leading and trailing whitespace
    data["Annotations"] = data["Annotations"].replace(
        r"^ +| +$", r"", regex=True
    )
    # Split the Label col, to see if there are multiple label categories
    pom = data["Annotations"].str.split("/", expand=True)
    # This means there were not multiple, so return the data
    if pom.shape[1] == 1:
        return data
    # Prepare column names for the multiple label columns
    if label_columns is None:
        label_columns = ["Label" + str(i + 1) for i in range(0, pom.shape[1])]
    pom.columns = label_columns
    data = data.drop("Annotations", axis=1)
    data = pd.concat([data, pom], axis=1)
    return data


def baseline_drift_filter(signal):
    """
    Remove baseline drift from signal using DC-blocker filter
    Reference: https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    """
    alpha = 0.999  # should be adjusted
    b = [1, -1]
    a = [1, -alpha]
    return scipy.signal.filtfilt(b, a, signal)


def filterr(signal):
    signal = np.array(signal)
    sig = scipy.signal.medfilt(signal)
    sig = bandpass_filter(sig, 3, [0.05, 0.5], 50)
    return sig


def get_segments(data, column):
    segments = segmentation.find_subsegments_indices(data[column])
    labels = segmentation.get_segment_label(data, segments, column)
    res = pd.DataFrame(
        [np.append(x, y) for x, y in zip(segments, labels)],
        columns=["start", "end", "label"],
    )
    res[["start", "end"]] = res[["start", "end"]].astype("int")
    return res


def draw_labels(data, ax):
    if "Position" in data.columns:
        ymin, ymax = ax.get_ylim()
        segments = get_segments(data, "Position")
        sitting_standing = segments.loc[
            segments["label"].str.contains("sitting-")
        ]
        standing_sitting = segments.loc[
            segments["label"].str.contains("standing-")
        ]

        # add lines for sitting to standing transition
        ax.vlines(sitting_standing["start"], ymin, ymax, colors="b", alpha=0.9)
        ax.vlines(sitting_standing["end"], ymin, ymax, colors="b", alpha=0.9)
        # add lines for standing to sitting transition
        ax.vlines(standing_sitting["start"], ymin, ymax, colors="r", alpha=0.9)
        ax.vlines(standing_sitting["end"], ymin, ymax, colors="r", alpha=0.9)


def plot_data(data, col, title, extremes, threshold, ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.plot(data.index, data[col], c="tab:blue")
    draw_labels(data, ax)
    # Add the peaks and valleys detected from the signal
    for type, extreme in extremes:
        ax.scatter(
            extreme, data[col].iloc[extreme], c=color_dict[type], alpha=0.7
        )

    ax.set_xticks(range(0, data.shape[0], 500))
    if threshold is not None:
        # draw the threshold used for finding the extremes
        ax.axhline(threshold)
        ax.axhline(-threshold)

    ax.set_title(title)
    return ax


def plot_data2(data, col, title, transitions, threshold, ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.plot(data.index, data[col], c="tab:blue")

    for transition in transitions:
        e1 = transition[0]
        e2 = transition[1]
        c = color_dict[e1[0]]
        # add lines for sitting to standing transition
        ax.axvspan(e1[1], e2[1], color=c, alpha=0.3)

        # Add the peaks and valleys detected from the signal
        ax.scatter(e1[1], data[col].iloc[e1[1]], c=c, alpha=0.7)
        ax.scatter(e2[1], data[col].iloc[e2[1]], c=c, alpha=0.7)

    ax.set_xticks(range(0, data.shape[0], 500))
    if threshold is not None:
        # draw the threshold used for finding the extremes
        ax.axhline(threshold)
        ax.axhline(-threshold)

    ax.set_title(title)
    return ax


def plot_data_ODR(data, col, title, extremes, threshold=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.plot(data.index, data[col], c="tab:blue")
    draw_labels(data, ax)

    for type, extreme in extremes:
        # Add the peaks and valleys detected from the signal
        ax.scatter(
            extreme, data[col].iloc[extreme], c=color_dict[type], alpha=0.7
        )
        # highlight the transitions
        ax.axvspan(
            extreme - 25, extreme + 25, color=color_dict[type], alpha=0.3
        )

    ax.set_xticks(range(0, data.shape[0], 500))
    if threshold is not None:
        # draw the threshold used for finding the extremes
        ax.axhline(threshold)
        ax.axhline(-threshold)

    ax.set_title(title)
    return ax


def find_peaks(signal):
    threshold = np.nanstd(signal) * 1.5
    peaks = peakutils.indexes(signal, threshold, min_dist=50, thres_abs=True)
    valleys = peakutils.indexes(
        signal * -1, threshold, min_dist=50, thres_abs=True
    )

    return peaks, valleys, threshold


def find_peaks_ODR(signal):
    threshold = np.nanstd(signal) * 1.5
    peaks, _ = scipy_find_peaks(signal, threshold, width=50)
    valleys, _ = scipy_find_peaks(signal * -1, threshold, width=50)

    return peaks, valleys, threshold


def filter_transition_extremes(extremes):
    if len(extremes) < 3:
        return

    # create an empty list which will be filled with the filtered extremes
    filtered_extremes = []
    # The first and last extreme cant be filtered since they have 1 neighbour
    filtered_extremes.append(extremes[0])

    # We should not expect there to be three extremes of same kind in a sequence
    for extreme1, extreme2, extreme3 in zip(
        extremes[:-2], extremes[1:-1], extremes[2:]
    ):
        if extreme1[0] == extreme2[0] == extreme3[0]:
            warnings.warn(
                "There are multiple consecutive peaks, or valleys."
                + " They will be processed and removed"
            )
        else:
            # Add the extreme (peak or valley) if they conform to the scheme
            filtered_extremes.append(extreme2)

    # TODO: Higher context for whether there should've been an inverse extreme
    # detected or if the middle exterme was just an error

    filtered_extremes.append(extremes[-1])

    return filtered_extremes


def filter_transition_extremes_ODR(extremes):
    # create an empty list which will be filled with the filtered extremes
    filtered_extremes = []
    # The first and last extreme cant be filtered since they have 1 neighbour
    filtered_extremes.append(extremes[0])

    # We should not expect there to be three extremes of same kind in a sequence
    for extreme1, extreme2 in zip(extremes[:-1], extremes[1:]):
        if extreme1[0] == extreme2[0]:
            warnings.warn(
                "There are multiple consecutive peaks, or valleys."
                + " They will be processed and removed"
            )
        else:
            # Add the extreme (peak or valley) if they conform to the scheme
            filtered_extremes.append(extreme2)

    return filtered_extremes


def check_transition_sequence(extremes):
    no_errors = True
    # Check if the extremes appear in the order that we expect them to
    prevprev = extremes[0]
    prev = extremes[1]
    for current in extremes[2:]:
        # if there is sitting-standing (peak->valley), the next transition
        # should be standing-sitting (valley->peak). Check if thats the sequence
        if (prevprev[0] == prev[0] and prev[0] == current[0]) or (
            prevprev[0] != prev[0] and prev[0] != current[0]
        ):
            warnings.warn(
                "There seems to be an error in the sequence of the "
                + "detected peaks"
            )
            no_errors = False

        # else:
        # if prevprev[0] == prev[0] and prev[0] != current[0]:
        # if prevprev[0] != prev[0] and prev[0] == current[0]:

        prevprev = prev
        prev = current

    return no_errors


def get_neighbouring_distances(neighbours):
    if len(neighbours) < 2:
        raise ValueError("The minimum amount of neighbours should be 2.")

    # Calculate the distances between neighbouring peaks
    distances = [y[1] - x[1] for y, x in zip(neighbours[1:], neighbours[:-1])]
    transition_extremes = set()

    # Check the closest neighbour for each peak
    if len(distances) == 1:
        transition_extremes.add((neighbours[0], neighbours[1]))
    if len(distances) == 2:
        tmp_idx = np.argmin(distances)
        transition_extremes.add((neighbours[tmp_idx], neighbours[tmp_idx + 1]))

    """
    If we have the extremes x, y, z, k; then the corresponding distances that
    we would get are d1 (x <-> y), d2 (y <-> z), and d3 (z <-> k). We later 
    analyze these distances to see which of them are closest, in order to pair 
    them and determine the type of transition. 
    sitting->standing or standing->sitting 
    """

    for idx, d2 in enumerate(distances[1:-1]):
        idx += 1  # adjust the value of the index, since we are sending a slice
        d1 = distances[idx - 1]
        d3 = distances[idx + 1]

        if d2 < d1 and d2 < d3:
            """
            If the only 'real' pair of extremes are the y and z, and the others
            are falsely detected, then we expect the distance between y and z
            to be the smallest.
            """
            transition_extremes.add((neighbours[idx], neighbours[idx + 1]))
        elif d2 > d1 and d2 > d3:
            """
            If otherwise x and y, and z and k are real pairs, then we expect the
            distance between y and z to be greater compared to the others
            """
            transition_extremes.add((neighbours[idx - 1], neighbours[idx]))
            transition_extremes.add((neighbours[idx + 1], neighbours[idx + 2]))
        elif d1 > d2 and d2 > d3:
            if idx + 2 <= len(distances):
                d4 = distances[idx + 2]
                if d3 > d4:
                    transition_extremes.add(
                        (neighbours[idx], neighbours[idx + 1])
                    )
                # The other case of d3 < d4 will be taken care of next iteration
        elif d1 < d2 and d2 < d3:
            if idx + 2 <= len(distances):
                d4 = distances[idx + 2]
                if d3 > d4:
                    transition_extremes.add(
                        (neighbours[idx], neighbours[idx + 1])
                    )
                # The other case of d3 < d4 will be taken care of next iteration

    return transition_extremes


def pair_extremes(peaks, valleys):
    sequence = [("p", x) for x in peaks]
    sequence.extend([("v", x) for x in valleys])
    sequence.sort(key=lambda x: x[1])

    return sequence


def pipeline(path, sensor_col="Pressure/Raw"):
    data, _ = load_data(path)
    data = fix_annotations(data, ["Position", "Speed", "Activity"])
    # create axes for the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20, 19))
    data[sensor_col].plot(title="Raw", ax=ax1)
    sig = bandpass_filter(np.array(data[sensor_col]), 4, [0.05, 0.5], 50)
    data["sig"] = scipy.signal.medfilt(sig, 99)

    peaks, valleys, threshold = find_peaks(data["sig"])
    transition_sequence = pair_extremes(peaks, valleys)
    filtered_transition_sequence = filter_transition_extremes(
        transition_sequence
    )
    check_transition_sequence(filtered_transition_sequence)
    transition_extremes = get_neighbouring_distances(
        filtered_transition_sequence
    )

    plot_data(data, "sig", path, filtered_transition_sequence, threshold, ax2)
    plot_data2(data, "sig", path, transition_extremes, threshold, ax3)
    plt.tight_layout()

    return data


def read_data(path):
    if path[-3:] == "csv":
        sensor_col = "Pressure"
        data = pd.read_csv(path)
        data[sensor_col] = data[sensor_col] * 0.0128
    elif path[-3:] == "txt":
        sensor_col = "Pressure/Raw"
        data, _ = load_data(path)
        data = fix_annotations(data, ["Position", "Speed", "Activity"])

    return data, sensor_col


def pipeline_ODR(path, sensor_col="Pressure"):
    data, sensor_col = read_data(path)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20, 19))
    # plot the raw pressure data
    data[sensor_col].plot(title="Raw", ax=ax1)

    # plot the medfilt, smooth processed signal
    medfilt = scipy.signal.medfilt(np.array(data[sensor_col]), 99)
    pd.DataFrame(medfilt).rolling(50).sum().plot(
        title="medfilt + 50 sum", ax=ax2
    )

    # process signal
    data["sig"] = bandpass_filter(
        data[sensor_col], order=2, fcritical=[0.001, 0.2], fs=50
    )
    data["sig"] = data["sig"].diff()

    peaks, valleys, threshold = find_peaks_ODR(data["sig"])
    extremes = pair_extremes(peaks, valleys)
    extremes = filter_transition_extremes_ODR(extremes)

    plot_data_ODR(data, "sig", f, extremes, None, ax=ax3)
    plt.tight_layout()
    plt.show()

    return data


if __name__ == "__main__":
    path = "./data/ODR.csv"
    pressure_col = "Pressure/Raw"

    # data = pipeline("./data/upstairs.txt")

    datasets = {}
    odr_datasets = {}
    for f in os.listdir(r"./data"):
        odr_datasets[f] = pipeline_ODR(os.path.join("./data", f))

        # if "ODR" in f:
        #     odr_datasets[f] = pipeline_ODR(os.path.join("./data", f))
        # else:
        #     datasets[f] = pipeline(os.path.join("./data", f))
