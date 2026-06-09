import numpy as np
import matplotlib.pyplot as plt


'''
The goal of this script is to create a graphical user interface, which will be used for data visualization and for manual correction of misidentified peaks.
'''


class ECGPeakEditor:

    def __init__(self, signal, peaks):

        self.signal = signal

        self.original_peaks = set(peaks)
        self.current_peaks = set(peaks)

        self.added_peaks = set()
        self.removed_peaks = set()

        self.window = 2000
        self.offset = 0

        self.snap_radius = 30
        self.remove_tolerance = 20

        self.fig, self.ax = plt.subplots()

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.onkey)

        self.draw()

    def draw(self):

        self.ax.clear()

        start = self.offset
        end = min(len(self.signal), start + self.window)

        x = np.arange(start, end)

        self.ax.plot(x, self.signal[start:end], linewidth=0.8)

        original_visible = [p for p in self.original_peaks if start <= p < end]
        added_visible = [p for p in self.added_peaks if start <= p < end]
        removed_visible = [p for p in self.removed_peaks if start <= p < end]

        if original_visible:
            self.ax.scatter(
                original_visible,
                self.signal[original_visible],
                color="darkorange",
                label="original"
            )

        if added_visible:
            self.ax.scatter(
                added_visible,
                self.signal[added_visible],
                color="lime",
                label="added"
            )

        if removed_visible:
            self.ax.scatter(
                removed_visible,
                self.signal[removed_visible],
                color="red",
                marker="x",
                label="removed"
            )

        self.ax.set_title(
            "Left click = add peak | Right click = remove peak | Scroll = zoom | Arrows = move | Enter = finish"
        )

        self.ax.set_xlim(start, end)

        self.ax.legend(loc="upper right")

        self.fig.canvas.draw()

    def find_local_peak(self, x):

        start = max(0, x - self.snap_radius)
        end = min(len(self.signal), x + self.snap_radius)

        local_segment = self.signal[start:end]

        if len(local_segment) == 0:
            return x

        local_max = np.argmax(local_segment)

        return start + local_max

    def onclick(self, event):

        if event.inaxes != self.ax:
            return

        x = int(event.xdata)

        # LEFT CLICK = ADD
        if event.button == 1:

            peak = self.find_local_peak(x)

            if peak in self.removed_peaks:
                self.removed_peaks.remove(peak)
                self.current_peaks.add(peak)

            elif peak not in self.current_peaks:
                self.current_peaks.add(peak)

                if peak not in self.original_peaks:
                    self.added_peaks.add(peak)

        # RIGHT CLICK = REMOVE
        elif event.button == 3:

            nearby = [p for p in self.current_peaks if abs(p - x) <= self.remove_tolerance]

            if not nearby:
                return

            peak = nearby[0]

            self.current_peaks.remove(peak)

            if peak in self.added_peaks:
                self.added_peaks.remove(peak)

            elif peak in self.original_peaks:
                self.removed_peaks.add(peak)

        self.draw()

    def onscroll(self, event):

        if event.button == "up":
            self.window = max(200, int(self.window * 0.8))

        elif event.button == "down":
            self.window = min(len(self.signal), int(self.window * 1.2))

        self.draw()

    def onkey(self, event):

        step = int(self.window * 0.2)

        if event.key == "right":
            self.offset = min(len(self.signal) - self.window, self.offset + step)

        elif event.key == "left":
            self.offset = max(0, self.offset - step)

        elif event.key == "up":
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(ymin * 0.5, ymax * 0.5)

        elif event.key == "down":
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(ymin * 1.5, ymax * 1.5)

        elif event.key == "enter":
            plt.close(self.fig)

        self.draw()

    def run(self):

        plt.show(block=True)

        final_peaks = np.array(sorted(self.current_peaks))
        added = np.array(sorted(self.added_peaks))
        removed = np.array(sorted(self.removed_peaks))

        return final_peaks, added, removed


def correct_ecg_peaks_gui(signal, peaks):

    editor = ECGPeakEditor(signal, peaks)

    return editor.run()


def plot_ecg_peak_correction(participant, block, trial,
    signal,
    original_peaks,
    added_peaks,
    removed_peaks,
    final_peaks=None,
    title=True
):

    plt.figure(figsize=(14,4))

    x = np.arange(len(signal))

    plt.plot(x, signal, linewidth=1)

    if len(original_peaks) > 0:
        plt.scatter(
            original_peaks,
            signal[original_peaks],
            color="blue",
            label="Original peaks"
        )


    if len(removed_peaks) > 0:
        plt.scatter(
            removed_peaks,
            signal[removed_peaks],
            color="black",
            marker="x",
            label="Removed peaks"
        )

    if final_peaks is not None and len(final_peaks) > 0:
        plt.scatter(
            final_peaks,
            signal[final_peaks],
            color="red",
            label="Final peaks"
        )

    if len(added_peaks) > 0:
        plt.scatter(
            added_peaks,
            signal[added_peaks],
            color="green",
            label="Added peaks"
        )

    plt.legend()

    plt.xlabel("Samples")
    plt.ylabel("ECG amplitude")

    if title:
        plt.title(f"{participant} | block {block} trial {trial}")

    plt.tight_layout()

    plt.savefig(f"{participant}_block{block}_trial{trial}_ECG_QC_plot.png", dpi=150)
    plt.show(block=True)
