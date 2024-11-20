import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.simpledialog import askfloat, askstring
from tkinter import Menu


def DerivativeSignal():
    return np.array([1.0 * i for i in range(1, 101)])
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

        self.continuous_plot_button = tk.Button(self.root, text="Plot Continuous Representation",
                                                command=self.plot_continuous_representation)
        self.continuous_plot_button.pack()

        self.discrete_plot_button = tk.Button(self.root, text="Plot Discrete Representation",
                                              command=self.plot_discrete_representation)
        self.discrete_plot_button.pack()

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit)
        self.exit_button.pack()

        self.canvas = None
        self.quantization_levels = None

        # Create a menu bar
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # Create a "Frequency Domain" menu
        self.frequency_domain_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Frequency Domain", menu=self.frequency_domain_menu)
        self.frequency_domain_menu.add_command(label="Apply Fourier Transform", command=self.apply_fourier_transform)
        self.frequency_domain_menu.add_command(label="Compute DCT", command=self.compute_dct)
        self.frequency_domain_menu.add_command(label="Remove DC Component", command=self.remove_dc_component)

        # Create a "Time Domain" menu
        self.time_domain_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Time Domain", menu=self.time_domain_menu)
        self.time_domain_menu.add_command(label="Smoothing", command=self.smoothing)
        self.time_domain_menu.add_command(label="Delay or advance Signal", command=self.delay_advance_signal)
        self.time_domain_menu.add_command(label="fold Signal", command=self.fold_signal)
        self.time_domain_menu.add_command(label="delay advance folded signal", command=self.delay_advance_folded_signal)
        self.time_domain_menu.add_command(label="remove dc component", command=self.remove_dc_component_frequency_domain)
        self.time_domain_menu.add_command(label="convolve signals", command=self.convolve_signals)
        self.time_domain_menu.add_command(label="Compute First Derivative", command=self.compute_first_derivative)
        self.time_domain_menu.add_command(label="Compute Second Derivative", command=self.compute_second_derivative)
    def load_signal(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    signal_data = {}

                    for line in lines:
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            coordinates = eval(parts[0])
                            value = float(parts[1])
                            signal_data[coordinates] = value
                        else:

                            values = line.strip().split()
                            if len(values) >= 2:
                                coordinates = tuple(map(int, values[:-1]))
                                value = float(values[-1])
                                signal_data[coordinates] = value

                    if signal_data:
                        if self.signal_1 is None:
                            self.signal_1 = np.array(list(signal_data.values()))
                        else:
                            self.signal_2 = np.array(list(signal_data.values()))
                        print("Signal loaded successfully.")
                    else:
                        print("No valid signal data found in the file.")

            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    def calcdft(self, xn, N):
        L = len(xn)
        if N < L:
            raise ValueError('N must be greater than or equal to L!!')

        x1 = np.concatenate((xn, np.zeros(N - L)))
        W = np.zeros((N, N), dtype=np.complex128)

        for k in range(N):
            for n in range(N):
                p = np.exp(-1j * 2 * np.pi * n * k / N)
                W[k, n] = p

        print('Transformation matrix for DFT:')
        print(W)

        Xk = np.dot(W, x1)

        return Xk

    def apply_fourier_transform(self):
        if self.signal_1 is not None:
            N = len(self.signal_1)

            Xk = self.calcdft(self.signal_1, N)


            print('DFT X(k): ')
            print(Xk)


            mgXk = np.abs(Xk)
            phaseXk = np.angle(Xk)
            k = np.arange(N)

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.stem(k, mgXk)
            plt.title('DFT sequence')
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')

            plt.subplot(2, 1, 2)
            plt.stem(k, phaseXk)
            plt.title('Phase of the DFT sequence')
            plt.xlabel('Frequency')
            plt.ylabel('Phase')

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

        else:
            print("No signal data to apply DFT. Generate or load a signal first.")

    def calcidft(self, Xk, N):
        if N != len(Xk):
            raise ValueError('Length of Xk must be equal to N!!')

        x1 = [0] * N

        for n in range(N):
            for k in range(N):
                p = np.exp(1j * 2 * np.pi * n * k / N)
                x1[n] += Xk[k] * p

            x1[n] /= N

        return x1

    def apply_idft(self):
        if self.signal_1 is not None:
            N = len(self.signal_1)

            # Assuming self.signal_1 contains the frequency-domain signal
            Xk = self.signal_1

            x1 = self.calcidft(Xk, N)

            print('iDFT x(n): ')
            print(x1)

            n = list(range(N))

            plt.figure(figsize=(12, 6))
            plt.stem(n, [x.real for x in x1])
            plt.title('iDFT sequence')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

        else:
            print("No signal data to apply iDFT. Generate or load a signal first.")


    def calcdct(self, signal):
        N = len(signal)
        Xk = np.zeros(N)

        for k in range(N):
            answer = 0
            for n in range(N):
                Seta = np.pi * (2 * n - 1) * (2 * k - 1)
                Seta /= (4 * N)
                angle = np.cos(Seta)
                answer += (signal[n] * angle)
            answer = answer * np.sqrt(2 / N)
            Xk[k] = answer

        return Xk

    def compute_dct(self):
        if self.signal_1 is not None:
            Xk_dct = self.calcdct(self.signal_1)


            print('DCT X(k): ')
            print(Xk_dct)


            k = np.arange(len(Xk_dct))

            plt.figure(figsize=(12, 6))
            plt.stem(k, Xk_dct)
            plt.title('DCT Coefficients')
            plt.xlabel('Coefficient Index')
            plt.ylabel('Amplitude')

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

            num_coefficients_to_save = askfloat("Save Coefficients", "Enter the number of coefficients to save:")
            if num_coefficients_to_save is None:
                return

            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                try:
                    with open(file_path, 'w') as file:
                        for i in range(int(num_coefficients_to_save)):
                            file.write(f"{i} {Xk_dct[i]}\n")
                    print(f"{int(num_coefficients_to_save)} DCT coefficients saved to '{file_path}'.")
                except Exception as e:
                    print(f"An error occurred while saving the coefficients: {str(e)}")
        else:
            print("No signal data to compute DCT. Generate or load a signal first.")

    def remove_dc_component(self):
        if self.signal_1 is not None:

            self.signal_1 = self.signal_1 - np.mean(self.signal_1)

            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Signal 1')
            if self.signal_2 is not None:
                plt.plot(self.signal_2, 'r-', label='Signal 2')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Signal after Removing DC Component')
            plt.legend()
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

            print("DC component removed successfully.")
        else:
            print("No signal data to remove DC component. Generate or load a signal first.")

    def smoothing(self):
        if self.signal_1 is not None:
            try:
                num_points = askfloat("Smoothing", "Enter the number of points for smoothing:")
                if num_points is None or num_points <= 0:
                    print("Invalid number of points for smoothing.")
                    return

                smoothed_signal = self.apply_smoothing(self.signal_1, int(num_points))

                plt.figure(figsize=(12, 6))
                plt.plot(self.signal_1, 'b-', label='Original Signal')
                plt.plot(smoothed_signal, 'r-', label=f'Smoothed Signal ({int(num_points)} points)')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.title('Signal Smoothing (Moving Average)')
                plt.legend()
                plt.grid()

                if self.canvas:
                    self.canvas.get_tk_widget().pack_forget()
                self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
                self.canvas.get_tk_widget().pack()

                print(f"Signal smoothed successfully with {int(num_points)} points.")
            except Exception as e:
                print(f"An error occurred during smoothing: {str(e)}")
        else:
            print("No signal data to smooth. Generate or load a signal first.")

    def apply_smoothing(self, signal, num_points):
        smoothed_signal = np.convolve(signal, np.ones(num_points) / num_points, mode='valid')
        return np.concatenate((signal[:num_points - 1], smoothed_signal))

    def delay_advance_signal(self):
        if self.signal_1 is not None:
            num_steps = askfloat("Delay/Advance Signal", "Enter the number of steps to delay/advance the signal:")
            if num_steps is None:
                return


            if num_steps > 0:
                delayed_signal = np.concatenate((np.zeros(int(num_steps)), self.signal_1[:-int(num_steps)]))
            elif num_steps < 0:
                advanced_signal = np.concatenate((self.signal_1[-int(num_steps):], np.zeros(-int(num_steps))))
            else:
                return  # No change

            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Original Signal')
            if num_steps > 0:
                plt.plot(delayed_signal, 'r-', label=f'Delayed Signal ({int(num_steps)} steps)')
            elif num_steps < 0:
                plt.plot(advanced_signal, 'g-', label=f'Advanced Signal ({-int(num_steps)} steps)')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Signal Delay/Advance')
            plt.legend()
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()
        else:
            print("No signal data to delay/advance. Generate or load a signal first.")

    def fold_signal(self):
        if self.signal_1 is not None:
            folded_signal = self.signal_1[::-1]

            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Original Signal')
            plt.plot(folded_signal, 'r-', label='Folded Signal')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Signal Folding')
            plt.legend()
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()
        else:
            print("No signal data to fold. Generate or load a signal first.")

    def delay_advance_folded_signal(self):
        if self.signal_1 is not None:

            folded_signal = self.signal_1[::-1]
            num_steps = askfloat("Delay/Advance Folded Signal",
                                 "Enter the number of steps to delay/advance the folded signal:")
            if num_steps is None:
                return


            if num_steps > 0:
                delayed_signal = np.concatenate((np.zeros(int(num_steps)), folded_signal[:-int(num_steps)]))
            elif num_steps < 0:
                advanced_signal = np.concatenate((folded_signal[-int(num_steps):], np.zeros(-int(num_steps))))
            else:
                return  # No change


            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Original Signal')
            plt.plot(folded_signal, 'r-', label='Folded Signal')
            if num_steps > 0:
                plt.plot(delayed_signal, 'g-', label=f'Delayed Folded Signal ({int(num_steps)} steps)')
            elif num_steps < 0:
                plt.plot(advanced_signal, 'y-', label=f'Advanced Folded Signal ({-int(num_steps)} steps)')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Signal Delay/Advance on Folded Signal')
            plt.legend()
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()
        else:
            print("No signal data to delay/advance the folded signal. Generate or load a signal first.")

    def remove_dc_component_frequency_domain(self):
        if self.signal_1 is not None:
            # Compute the DFT of the signal
            N = len(self.signal_1)
            Xk = self.calcdft(self.signal_1, N)
            Xk[0] = 0
            modified_signal = self.calcidft(Xk, N)

            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Original Signal')
            plt.plot(modified_signal.real, 'r-', label='Signal after Removing DC Component in Frequency Domain')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Signal without DC Component in Frequency Domain')
            plt.legend()
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()
        else:
            print("No signal data to remove DC component. Generate or load a signal first.")

    def convolution(self, signal1, signal2):
        len1 = len(signal1)
        len2 = len(signal2)
        result = [0] * (len1 + len2 - 1)

        for i in range(len1):
            for j in range(len2):
                result[i + j] += signal1[i] * signal2[j]

        return result

    def convolve_signals(self):
        if self.signal_1 is not None and self.signal_2 is not None:
            convolved_signal = self.convolution(self.signal_1, self.signal_2)

            plt.figure(figsize=(12, 6))
            plt.plot(self.signal_1, 'b-', label='Signal 1')
            plt.plot(self.signal_2, 'r-', label='Signal 2')
            plt.plot(convolved_signal, 'g-', label='Convolved Signal')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Signal Convolution')
            plt.legend()
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()
        else:
            print("Not enough signal data to convolve. Generate or load both signals first.")

    def compute_first_derivative(self):
        input_signal = DerivativeSignal()
        N = len(input_signal)

        # Assuming self.signal_1 contains the frequency-domain signal
        Xk = self.signal_1

        # Compute the inverse DFT to get the time-domain signal
        first_derivative_signal = self.calcidft(1j * np.arange(N) * 2 * np.pi / N * Xk, N).real

        plt.figure(figsize=(12, 6))
        plt.plot(input_signal, 'b-', label='Original Signal')
        plt.plot(first_derivative_signal, 'r-', label='First Derivative')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('First Derivative of the Signal')
        plt.legend()
        plt.grid()

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
        self.canvas.get_tk_widget().pack()

    def compute_second_derivative(self):
        input_signal = DerivativeSignal()
        N = len(input_signal)

        # Assuming self.signal_1 contains the frequency-domain signal
        Xk = self.signal_1

        # Compute the inverse DFT to get the time-domain signal
        second_derivative_signal = self.calcidft(-(1j * np.arange(N) * 2 * np.pi / N)**2 * Xk, N).real

        plt.figure(figsize=(12, 6))
        plt.plot(input_signal, 'b-', label='Original Signal')
        plt.plot(second_derivative_signal, 'g-', label='Second Derivative')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Second Derivative of the Signal')
        plt.legend()
        plt.grid()

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
        self.canvas.get_tk_widget().pack()


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