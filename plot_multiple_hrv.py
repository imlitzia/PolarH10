"""Plot multiple HRV recordings from zero seconds and calculate their average."""

from pathlib import Path
from datetime import datetime
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    # Some Polar filenames contain U+202F, which Windows cp1252 cannot print.
    sys.stdout.reconfigure(errors="replace")


TIME_COLUMN = "time"
VALUE_COLUMN = "HRV_LFHF"
PROGRAM_VERSION = "2026-08-31-v6"
ECG_COLUMNS = ("ecg", "ECG")
RAW_TIME_COLUMNS = ("time", "Time", "UNIX Timestamp", "timestamp")
ECG_SAMPLING_RATE = 130
HRV_WINDOW_SAMPLES = 5000
# One calculation per second. A 10-sample step creates thousands of expensive,
# almost-identical NeuroKit calculations and can terminate Python from resource use.
HRV_STEP_SAMPLES = ECG_SAMPLING_RATE


def find_processed_hrv_file(raw_filepath):
    """Find the newest versioned HRV result corresponding to a selected raw CSV."""
    raw_stem = Path(raw_filepath).stem
    expected_names = {
        f"{raw_stem}_with_hrv.csv".casefold(),
        f"{raw_stem}_timeadjusted_with_hrv.csv".casefold(),
    }
    if raw_stem.casefold().endswith("_timeadjusted"):
        expected_names.add(f"{raw_stem}_with_hrv.csv".casefold())

    matches = []
    for version_directory in Path.cwd().glob("V_*"):
        if not version_directory.is_dir():
            continue
        for candidate in version_directory.glob("*.csv"):
            if candidate.name.casefold() in expected_names:
                matches.append(candidate)

    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def find_latest_processed_files():
    """Return all HRV results from the newest version folder that has them."""
    version_directories = []
    for directory in Path.cwd().glob("V_*"):
        if not directory.is_dir():
            continue
        try:
            version_number = int(directory.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        version_directories.append((version_number, directory))

    for _version, directory in sorted(version_directories, reverse=True):
        files = sorted(directory.glob("*_with_hrv.csv"))
        if files:
            return files
    return []


def timestamp_scale(values):
    """Return the divisor that converts Unix timestamp differences to seconds."""
    magnitude = abs(float(values.iloc[0]))
    if magnitude >= 1e17:
        return 1e9  # nanoseconds
    if magnitude >= 1e14:
        return 1e6  # microseconds
    if magnitude >= 1e11:
        return 1e3  # milliseconds
    return 1.0


def calculate_hrv_from_raw(frame, filename):
    """Calculate LF/HF values and elapsed times from a raw Polar ECG CSV."""
    import neurokit2 as nk

    ecg_column = next((column for column in ECG_COLUMNS if column in frame.columns), None)
    if ecg_column is None:
        raise ValueError(
            f"needs either '{VALUE_COLUMN}' data or a raw ECG/ecg column"
        )

    ecg = pd.to_numeric(frame[ecg_column], errors="coerce").interpolate(limit_direction="both")
    if ecg.isna().all():
        raise ValueError(f"column '{ecg_column}' contains no numeric ECG values")
    if len(ecg) < HRV_WINDOW_SAMPLES:
        raise ValueError(
            f"has {len(ecg)} ECG samples; at least {HRV_WINDOW_SAMPLES} are required"
        )

    time_column = next((column for column in RAW_TIME_COLUMNS if column in frame.columns), None)
    raw_times = None
    time_divisor = None
    if time_column is not None:
        raw_times = pd.to_numeric(frame[time_column], errors="coerce")
        if not raw_times.isna().all():
            raw_times = raw_times.interpolate(limit_direction="both")
            time_divisor = timestamp_scale(raw_times)

    elapsed_times = []
    lf_hf_values = []
    starts = range(0, len(ecg) - HRV_WINDOW_SAMPLES + 1, HRV_STEP_SAMPLES)
    total_windows = ((len(ecg) - HRV_WINDOW_SAMPLES) // HRV_STEP_SAMPLES) + 1

    for window_number, start in enumerate(starts, start=1):
        if window_number == 1 or window_number % 500 == 0 or window_number == total_windows:
            print(f"Calculating HRV for {filename}: {window_number}/{total_windows}")

        window = ecg.iloc[start : start + HRV_WINDOW_SAMPLES].to_numpy()
        try:
            clean = nk.ecg_clean(window, sampling_rate=ECG_SAMPLING_RATE)
            _signals, peaks = nk.ecg_peaks(clean, sampling_rate=ECG_SAMPLING_RATE)
            hrv = nk.hrv_frequency(peaks, sampling_rate=ECG_SAMPLING_RATE)
            value = pd.to_numeric(hrv[VALUE_COLUMN], errors="coerce").iloc[0]
        except Exception as error:
            print(f"Skipping HRV window {window_number} in {filename}: {error}")
            continue

        if pd.isna(value):
            continue
        if raw_times is not None:
            elapsed = (raw_times.iloc[start] - raw_times.iloc[0]) / time_divisor
        else:
            elapsed = start / ECG_SAMPLING_RATE
        elapsed_times.append(float(elapsed))
        lf_hf_values.append(float(value))

    if not elapsed_times:
        raise ValueError("HRV calculation produced no valid LF/HF values")
    return np.asarray(elapsed_times), np.asarray(lf_hf_values)


def load_hrv_file(filepath):
    """Load one HRV CSV and return clean elapsed-time and LF/HF arrays."""
    frame = pd.read_csv(filepath)
    if TIME_COLUMN not in frame.columns or VALUE_COLUMN not in frame.columns:
        processed_file = find_processed_hrv_file(filepath)
        if processed_file is not None:
            print(
                f"Using existing HRV result for {Path(filepath).name}: "
                f"{processed_file}",
                flush=True,
            )
            frame = pd.read_csv(processed_file)
        else:
            print(
                f"No processed HRV result found for {Path(filepath).name}; "
                "calculating from raw ECG at one point per second.",
                flush=True,
            )
            return calculate_hrv_from_raw(frame, Path(filepath).name)

    if TIME_COLUMN not in frame.columns or VALUE_COLUMN not in frame.columns:
        return calculate_hrv_from_raw(frame, Path(filepath).name)

    data = frame[[TIME_COLUMN, VALUE_COLUMN]].copy()
    data[TIME_COLUMN] = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors="coerce")
    data = data.dropna().sort_values(TIME_COLUMN)

    if data.empty:
        raise ValueError("contains no valid time and HRV_LFHF values")

    # Average duplicate timestamps so interpolation receives a unique x-axis.
    data = data.groupby(TIME_COLUMN, as_index=False)[VALUE_COLUMN].mean()
    data[TIME_COLUMN] -= data[TIME_COLUMN].iloc[0]
    return data[TIME_COLUMN].to_numpy(), data[VALUE_COLUMN].to_numpy()


def choose_grid_step(recordings):
    """Use the typical sampling interval while avoiding an excessively dense grid."""
    intervals = []
    for times, _values, _label in recordings:
        positive_differences = np.diff(times)
        positive_differences = positive_differences[positive_differences > 0]
        if positive_differences.size:
            intervals.append(float(np.median(positive_differences)))

    if not intervals:
        return 1.0
    return max(min(intervals), 0.01)


def calculate_average(recordings):
    """Average recordings only over the duration shared by every file."""
    step = choose_grid_step(recordings)
    shortest_duration = min(times[-1] for times, _values, _label in recordings)
    common_time = np.arange(0.0, shortest_duration + step / 2, step)
    common_time = common_time[common_time <= shortest_duration]

    interpolated = []
    for times, values, _label in recordings:
        interpolated.append(
            np.interp(common_time, times, values, left=np.nan, right=np.nan)
        )

    value_matrix = np.vstack(interpolated)
    valid_counts = np.sum(~np.isnan(value_matrix), axis=0)
    totals = np.nansum(value_matrix, axis=0)
    average = np.divide(
        totals,
        valid_counts,
        out=np.full(totals.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    return common_time, average, valid_counts


def plot_recordings(recordings, average_time, average_values):
    """Create the combined individual-recording and average trend graph."""
    figure, axis = plt.subplots(figsize=(12, 7))
    shared_end_time = average_time[-1]

    for times, values, label in recordings:
        shared_range = times <= shared_end_time
        axis.plot(
            times[shared_range],
            values[shared_range],
            linewidth=1,
            alpha=0.55,
            label=label,
        )

    axis.plot(
        average_time,
        average_values,
        color="black",
        linewidth=3,
        alpha=0.6,
        label=f"Average ({len(recordings)} files)",
        zorder=10,
    )
    axis.set_title("HRV LF/HF Trends Aligned at Start")
    axis.set_xlabel("Elapsed Time (seconds)")
    axis.set_ylabel("LF/HF Ratio")
    axis.set_xlim(0, shared_end_time)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def main():
    print(f"Multi-HRV Plotter {PROGRAM_VERSION}", flush=True)
    print(f"Running: {Path(__file__).resolve()}", flush=True)
    print(f"Raw HRV step: {HRV_STEP_SAMPLES} samples", flush=True)

    # Tk file dialogs caused the Windows process to terminate when plotting on
    # this machine. Accept explicit paths, or use the newest processed folder.
    files = [Path(argument) for argument in sys.argv[1:]]
    if not files:
        files = find_latest_processed_files()
        if not files:
            raise FileNotFoundError("No processed *_with_hrv.csv files were found.")
        print(
            f"No file paths supplied; automatically using {len(files)} "
            f"processed files from {Path(files[0]).parent}.",
            flush=True,
        )

    print("Files to plot:", flush=True)
    for filepath in files:
        print(f"  {str(Path(filepath).resolve()).replace(chr(0x202f), ' ')}", flush=True)

    recordings = []
    errors = []
    for filepath in files:
        display_name = Path(filepath).name.replace("\u202f", " ")
        print(f"Loading: {display_name}", flush=True)
        try:
            times, values = load_hrv_file(filepath)
            plot_label = Path(filepath).stem.replace("\u202f", " ")
            recordings.append((times, values, plot_label))
            print(f"Loaded {len(values)} HRV points", flush=True)
        except Exception as error:
            errors.append(f"{display_name}: {error}")

    if not recordings:
        raise ValueError("None of the files could be plotted:\n" + "\n".join(errors))

    print("Calculating average trend...", flush=True)
    average_time, average_values, valid_counts = calculate_average(recordings)
    print("Rendering combined graph...", flush=True)
    figure = plot_recordings(recordings, average_time, average_values)

    # Save automatically so a hidden or blocked second file dialog cannot cause
    # the completed calculation to be lost.
    output_directory = Path.cwd() / "multi_hrv_output"
    output_directory.mkdir(exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_directory / f"combined_hrv_average_{run_timestamp}.png"
    csv_path = output_directory / f"combined_hrv_average_{run_timestamp}.csv"

    print("Saving combined graph...", flush=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    pd.DataFrame(
        {
            "elapsed_seconds": average_time,
            "average_HRV_LFHF": average_values,
            "files_in_average": valid_counts,
        }
    ).to_csv(csv_path, index=False)

    print(f"Plot saved to: {output_path.resolve()}", flush=True)
    print(f"Average data saved to: {csv_path.resolve()}", flush=True)

    if errors:
        print("Skipped files:", flush=True)
        for error in errors:
            print(f"  {error}", flush=True)
    plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Keep unexpected failures visible in the terminal instead of exiting
        # without explaining why no output appeared.
        traceback.print_exc()
        input("The program encountered an error. Press Enter to close...")
