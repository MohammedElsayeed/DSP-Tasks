import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.simpledialog import askfloat, askstring
from tkinter import Menu

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

        self.generate_cosine_button = tk.Button(self.root, text="Generate Cosine Wave",
                                                command=self.generate_cosine_wave)
        self.generate_cosine_button.pack()

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
        self.frequency_domain_menu.add_command(label="Modify Amplitude and Phase", command=self.modify_amplitude_phase)
        self.frequency_domain_menu.add_command(label="Reconstruct Signal", command=self.reconstruct_signal)
        self.frequency_domain_menu.add_command(label="Save Components", command=self.save_components)
        self.frequency_domain_menu.add_command(label="Load Components", command=self.load_components)
        self.frequency_domain_menu.add_command(label="Compute DCT", command=self.compute_dct)
        self.frequency_domain_menu.add_command(label="Remove DC Component", command=self.remove_dc_component)

    # ... (your existing code)

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
                            coordinates = eval(parts[0])  # Using eval to parse the tuple
                            value = float(parts[1])
                            signal_data[coordinates] = value
                        else:
                            # Handle the case where a line contains both coordinates and a value
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

            # Display DFT results
            print('DFT X(k): ')
            print(Xk)

            # Plot magnitude and phase
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

        W = np.zeros((N, N), dtype=np.complex128)

        for k in range(N):
            for n in range(N):
                p = np.exp(1j * 2 * np.pi * n * k / N)
                W[k, n] = p

        print('Transformation matrix for iDFT:')
        print(W)

        x1 = np.dot(W.conjugate().transpose(), Xk) / N

        return x1

    def apply_idft(self):
        if self.signal_1 is not None:
            N = len(self.signal_1)

            # Assume you have the Xk values (complex numbers) from somewhere
            # For simplicity, let's use the DFT results obtained in the apply_dft function
            Xk = np.fft.fft(self.signal_1)

            x1 = self.calcidft(Xk, N)

            # Display iDFT result
            print('iDFT x(n): ')
            print(x1)

            # Plot the iDFT result
            n = np.arange(N)

            plt.figure(figsize=(12, 6))
            plt.stem(n, x1.real)
            plt.title('iDFT sequence')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

        else:
            print("No signal data to apply iDFT. Generate or load a signal first.")

    def modify_amplitude_phase(self):
        if self.signal_1 is not None:
            N = len(self.signal_1)

            # Compute the DFT of the signal
            Xk = self.calcdft(self.signal_1, N)

            # Display DFT results
            print('DFT X(k): ')
            print(Xk)

            # Ask the user for the frequency component to modify
            frequency_values = np.fft.fftfreq(len(self.signal_1))
            freq_to_modify = askfloat("Frequency Component", "Enter the frequency component (Hz) to modify:")
            if freq_to_modify is None:
                return

            # Find the nearest frequency component
            idx = np.argmin(np.abs(frequency_values - freq_to_modify))
            freq_to_modify = frequency_values[idx]

            # Ask the user for the new amplitude and phase
            new_amplitude = askfloat("Amplitude", f"Enter the new amplitude for {freq_to_modify} Hz:")
            if new_amplitude is None:
                return
            new_phase = askfloat("Phase", f"Enter the new phase (radians) for {freq_to_modify} Hz:")
            if new_phase is None:
                return

            # Modify the amplitude and phase in the frequency domain
            Xk[idx] = new_amplitude * np.exp(1j * new_phase)

            # Compute the inverse DFT to get the modified signal
            modified_signal = self.calcidft(Xk, N)

            # Update the signal with the modified component
            self.signal_1 = modified_signal.real

            # Plot and display the modified signal
            plt.figure(figsize=(12, 6))
            plt.stem(self.signal_1, linefmt='b-', markerfmt='bo', basefmt=' ')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Modified Signal (Discrete Representation)')
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

            print(f"Amplitude and phase for {freq_to_modify} Hz modified successfully.")
        else:
            print("No signal data to modify. Generate or load a signal first.")

    def reconstruct_signal(self):
        if self.signal_1 is not None:
            # Ask the user for the number of frequency components to reconstruct
            num_components = askfloat("Reconstruction", "Enter the number of frequency components to reconstruct:")
            if num_components is None:
                return

            # Initialize the reconstructed signal
            reconstructed_signal = np.zeros(len(self.signal_1))

            # Prompt the user for all frequency components at once
            components = []
            for _ in range(int(num_components)):
                component_data = askstring("Reconstruction", "Enter the component data (frequency, amplitude, phase):")
                if component_data is None:
                    return
                freq_to_add, amplitude_to_add, phase_to_add = map(float, component_data.split())
                components.append((freq_to_add, amplitude_to_add, phase_to_add))

            # Add the specified frequency components to the reconstructed signal
            for freq_to_add, amplitude_to_add, phase_to_add in components:
                N = len(self.signal_1)
                t = np.arange(N)
                component_signal = amplitude_to_add * np.exp(1j * phase_to_add) * np.exp(
                    2j * np.pi * freq_to_add * t / N)
                reconstructed_signal += component_signal

            # Plot and display the reconstructed signal immediately after user input
            plt.figure(figsize=(12, 6))
            plt.plot(reconstructed_signal, 'b-')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Reconstructed Signal (Continuous Representation)')
            plt.grid()

            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            self.canvas.get_tk_widget().pack()

            print("Signal reconstructed successfully.")
        else:
            print("No signal data to reconstruct. Generate or load a signal first.")

    def save_components(self):
        if self.signal_1 is not None:
            # Ask the user for the file name to save the components
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                try:
                    with open(file_path, 'w') as file:
                        # Write the components in polar form (amplitude and phase)
                        frequency_values = np.fft.fftfreq(len(self.signal_1))
                        fourier_transform = np.fft.fft(self.signal_1)
                        for freq, amplitude, phase in zip(frequency_values, np.abs(fourier_transform),
                                                          np.angle(fourier_transform)):
                            file.write(f"{freq} {amplitude} {phase}\n")
                    print(f"Frequency components saved to '{file_path}'.")
                except Exception as e:
                    print(f"An error occurred while saving the components: {str(e)}")
        else:
            print("No signal data to save components. Generate or load a signal first.")

    def load_components(self):
        # Ask the user to select a text file containing frequency components in polar form
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    frequency_values = []
                    amplitudes = []
                    phases = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            frequency_values.append(float(parts[0]))
                            amplitudes.append(float(parts[1]))
                            phases.append(float(parts[2]))
                    if frequency_values and amplitudes and phases:
                        reconstructed_signal = self.reconstruct_signal_from_components(frequency_values, amplitudes,
                                                                                       phases)
                        self.signal_1 = reconstructed_signal
                        print(f"Frequency components loaded and signal reconstructed successfully.")
                    else:
                        print("No valid frequency components found in the file.")
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    def reconstruct_signal_from_components(self, frequency_values, amplitudes, phases):
        num_samples = len(self.signal_1) if self.signal_1 is not None else 1024  # Default to 1024 samples
        t = np.arange(num_samples)
        reconstructed_signal = np.zeros(num_samples, dtype=complex)

        for freq, amplitude, phase in zip(frequency_values, amplitudes, phases):
            component_signal = amplitude * np.exp(1j * phase) * np.exp(2j * np.pi * freq * t / num_samples)
            reconstructed_signal += component_signal

        return reconstructed_signal.real

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

            # Display DCT results
            print('DCT X(k): ')
            print(Xk_dct)

            # Plot DCT coefficients
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

            # Ask the user for the number of coefficients to save
            num_coefficients_to_save = askfloat("Save Coefficients", "Enter the number of coefficients to save:")
            if num_coefficients_to_save is None:
                return

            # Save the first m coefficients to a text file
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
            # Subtract the mean value to remove the DC component
            self.signal_1 = self.signal_1 - np.mean(self.signal_1)

            # Plot and display the signal after removing the DC component
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