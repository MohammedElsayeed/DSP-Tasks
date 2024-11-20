import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.simpledialog import askfloat

class SignalProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")

        self.signal_1 = None
        self.signal_2 = None

        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self.root, text="Load Signal from File", command=self.load_signal)
        self.load_button.pack()

        self.generate_sine_button = tk.Button(self.root, text="Generate Sine Wave", command=self.generate_sine_wave)
        self.generate_sine_button.pack()

        self.generate_cosine_button = tk.Button(self.root, text="Generate Cosine Wave", command=self.generate_cosine_wave)
        self.generate_cosine_button.pack()

        self.continuous_plot_button = tk.Button(self.root, text="Plot Continuous Representation", command=self.plot_continuous_representation)
        self.continuous_plot_button.pack()

        self.discrete_plot_button = tk.Button(self.root, text="Plot Discrete Representation", command=self.plot_discrete_representation)
        self.discrete_plot_button.pack()

        self.addition_button = tk.Button(self.root, text="Add Signals", command=self.add_signals)
        self.addition_button.pack()

        self.subtract_button = tk.Button(self.root, text="Subtract Signals", command=self.subtract_signals)
        self.subtract_button.pack()

        self.multiply_button = tk.Button(self.root, text="Multiply Signal", command=self.multiply_signal)
        self.multiply_button.pack()

        self.square_button = tk.Button(self.root, text="Square Signal", command=self.square_signal)
        self.square_button.pack()

        self.shift_button = tk.Button(self.root, text="Shift Signal", command=self.shift_signal)
        self.shift_button.pack()

        self.normalize_button = tk.Button(self.root, text="Normalize Signal", command=self.normalize_signal)
        self.normalize_button.pack()

        self.accumulate_button = tk.Button(self.root, text="Accumulate Signal", command=self.accumulate_signal)
        self.accumulate_button.pack()

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit)
        self.exit_button.pack()

        self.canvas = None

    def load_signal(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    samples = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            samples.append(float(parts[1]))
                    if samples:
                        if self.signal_1 is None:
                            self.signal_1 = np.array(samples)
                        else:
                            self.signal_2 = np.array(samples)
                        print("Signal loaded successfully.")
                    else:
                        print("No valid signal data found in the file.")
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    def generate_sine_wave(self):
        amplitude = askfloat("Amplitude", "Enter amplitude:")
        if amplitude is None:
            return
        frequency = askfloat("Frequency", "Enter analog frequency (Hz):")
        if frequency is None:
            return
        phase_shift = askfloat("Phase Shift", "Enter phase shift (radians):")
        if phase_shift is None:
            return
        sampling_frequency = askfloat("Sampling Frequency", "Enter sampling frequency (Hz):")
        if sampling_frequency is None:
            return
        duration = askfloat("Duration", "Enter signal duration (seconds):")
        if duration is None:
            return

        num_samples = int(sampling_frequency * duration)  # Ensure the same number of samples for both signals
        t = np.arange(0, duration, 1 / sampling_frequency)
        if self.signal_1 is None:
            self.signal_1 = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
        else:
            self.signal_2 = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)

        print("Sine wave generated successfully.")

    def generate_cosine_wave(self):
        amplitude = askfloat("Amplitude", "Enter amplitude:")
        if amplitude is None:
            return
        frequency = askfloat("Frequency", "Enter analog frequency (Hz):")
        if frequency is None:
            return
        phase_shift = askfloat("Phase Shift", "Enter phase shift (radians):")
        if phase_shift is None:
            return
        sampling_frequency = askfloat("Sampling Frequency", "Enter sampling frequency (Hz):")
        if sampling_frequency is None:
            return
        duration = askfloat("Duration", "Enter signal duration (seconds):")
        if duration is None:
            return

        num_samples = int(sampling_frequency * duration)  # Ensure the same number of samples for both signals
        t = np.arange(0, duration, 1 / sampling_frequency)
        if self.signal_1 is None:
            self.signal_1 = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift)
        else:
            self.signal_2 = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift)

        print("Cosine wave generated successfully.")

    def add_signals(self):
        if self.signal_1 is not None and self.signal_2 is not None:
            if len(self.signal_1) == len(self.signal_2):
                self.signal_1 = self.signal_1 + self.signal_2
                self.signal_2 = None
                print("Signals added successfully.")
            else:
                print("Both signals must have the same length for addition.")
        else:
            print("Both signals must be generated or loaded for addition.")

    def subtract_signals(self):
        if self.signal_1 is not None and self.signal_2 is not None:
            if len(self.signal_1) == len(self.signal_2):
                self.signal_1 = self.signal_1 - self.signal_2
                self.signal_2 = None
                print("Signals subtracted successfully.")
            else:
                print("Both signals must have the same length for subtraction.")
        else:
            print("Both signals must be generated or loaded for subtraction.")

    def multiply_signal(self):
        constant = askfloat("Constant", "Enter the constant value:")
        if constant is None:
            return

        if self.signal_1 is not None:
            self.signal_1 = self.signal_1 * constant
            print(f"Signal multiplied by {constant} successfully.")
        else:
            print("No signal data to multiply. Generate or load a signal first.")

    def square_signal(self):
        if self.signal_1 is not None:
            self.signal_1 = np.square(self.signal_1)
            print("Signal squared successfully.")
        else:
            print("No signal data to square. Generate or load a signal first.")

    def shift_signal(self):
        constant = askfloat("Constant", "Enter the shifting constant (positive or negative):")
        if constant is None:
            return

        if self.signal_1 is not None:
            self.signal_1 = self.signal_1 + constant
            print(f"Signal shifted by {constant} successfully.")
        else:
            print("No signal data to shift. Generate or load a signal first.")

    def normalize_signal(self):
        range_option = askfloat("Range Option", "Enter the desired range (0 for 0 to 1, -1 for -1 to 1):")
        if range_option is None:
            return

        if range_option == 0:
            # Normalize to the range [0, 1]
            if self.signal_1 is not None:
                min_val = np.min(self.signal_1)
                max_val = np.max(self.signal_1)
                self.signal_1 = (self.signal_1 - min_val) / (max_val - min_val)
                print("Signal normalized to the range [0, 1] successfully.")
            else:
                print("No signal data to normalize. Generate or load a signal first.")
        elif range_option == -1:
            # Normalize to the range [-1, 1]
            if self.signal_1 is not None:
                min_val = np.min(self.signal_1)
                max_val = np.max(self.signal_1)
                self.signal_1 = (2 * (self.signal_1 - min_val) / (max_val - min_val)) - 1
                print("Signal normalized to the range [-1, 1] successfully.")
            else:
                print("No signal data to normalize. Generate or load a signal first.")
        else:
            print("Invalid range option. Enter 0 for [0, 1] or -1 for [-1, 1].")

    def accumulate_signal(self):
        if self.signal_1 is not None:
            self.signal_1 = np.cumsum(self.signal_1)
            print("Signal accumulated successfully.")
        else:
            print("No signal data to accumulate. Generate or load a signal first.")

    def plot_continuous_representation(self):
        if self.signal_1 is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Signal 1')
            if self.signal_2 is not None:
                plt.plot(self.signal_2, 'r-', label='Signal 2')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Continuous Representation of the Signal')
            plt.legend()
            plt.grid()
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()
        else:
            print("No signal data to plot. Generate a signal first.")

    def plot_discrete_representation(self):
        if self.signal_1 is not None:
            plt.figure(figsize=(12, 6))
            plt.stem(self.signal_1, linefmt='b-', markerfmt='bo', basefmt=' ')
            if self.signal_2 is not None:
                plt.stem(self.signal_2, linefmt='r-', markerfmt='ro', basefmt=' ')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Discrete Representation of the Signal')
            plt.grid()
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessingGUI(root)
    root.mainloop()
