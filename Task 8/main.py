import cmath
import tkinter as tk
def read_signal_data(filename):
    list1 = []
    list2 = []

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        value1, value2 = map(float, parts)
                        list1.append(value1)
                        list2.append(value2)
                    except ValueError:
                        print(f"Skipping invalid data in '{filename}': {line}")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")

    return list1, list2

x1,y1=read_signal_data("Corr_input signal1.txt")
x2,y2=read_signal_data("Corr_input signal2.txt")

x3,y3=read_signal_data("Input_conv_Sig1.txt")
x4,y4=read_signal_data("Input_conv_Sig2.txt")

def dft(x):
    N = len(x)
    result = [0] * N
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += x[n] * cmath.exp(-2j * cmath.pi * k * n / N)
        result[k] = sum_val
    return result

def idft(X):
    N = len(X)
    result = [0] * N
    for n in range(N):
        sum_val = 0
        for k in range(N):
            sum_val += X[k] * cmath.exp(2j * cmath.pi * k * n / N)
        result[n] = sum_val / N
    return result

#cross

def fast_convolution(signal1, signal2):
    N1 = len(signal1)
    N2 = len(signal2)

    signal1 += [0] * (N2 - 1)
    signal2 += [0] * (N1 - 1)

    spectrum1 = dft(signal1)
    spectrum2 = dft(signal2)

    # Multiply
    result_spectrum = [a * b for a, b in zip(spectrum1, spectrum2)]

    # Perform IDFT
    convolution_result = idft(result_spectrum)

    # Round the real and imaginary part
    convolution_result = [round(val.real) for val in convolution_result]

    return convolution_result


def fast_correlation(signal1, signal2):
    N = len(signal1)

    # Perform DFT
    dft_signal1 = dft(signal1)
    dft_signal2 = dft(signal2)

    # Take the complex conjugate of the first signal
    conjugate_signal1 = [complex(x.real, -x.imag) for x in dft_signal1]

    # Multiply
    correlation_result = [x * y for x, y in zip(dft_signal2, conjugate_signal1)]

    # In case of cross-correlation, divide by N
    if signal1 is not signal2:
        correlation_result = [x / N for x in correlation_result]

    # Perform IDFT
    correlation_result = idft(correlation_result)

    return [round(x.real,1) for x in correlation_result]


def do_fast_conv():
    result = fast_convolution(y3,y4)
    print("Convolution Result =",result)



def do_fast_corr():
    result = fast_correlation(y1,y2)
    print("Correlation Result =",result)


root = tk.Tk()
root.title("TASK 8")

corr = tk.Button(root, text="Fast Correlation ", command=do_fast_corr)
corr.pack()
conv = tk.Button(root, text="Fast Convolution", command=do_fast_conv)
conv.pack()

root.mainloop()




