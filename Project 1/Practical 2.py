import os
from fractions import Fraction
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")
def remove_dc(signal):
    dc_removed_signal = signal - np.mean(signal)
    return dc_removed_signal
def normalize(signal):
    normalized_signal = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
    return normalized_signal

def convlution(y, y1):
    conv_length = len(y) + len(y1) - 1
    y = np.pad(y, (0, conv_length - len(y)), 'constant')
    y1 = np.pad(y1, (0, conv_length - len(y1)), 'constant')
    # print(conv_length)

    complexreal, compleximag, complexreal1, compleximag1 = [], [], [], []
    for n in range(conv_length):
        real = 0.0
        imag = 0.0
        for k in range(conv_length):
            amplitude = y[k]
            phase = np.pi * 2 * n * k / conv_length
            real += amplitude * np.cos(phase)
            imag += (-amplitude * np.sin(phase))
        complexreal.append(real)
        compleximag.append(imag)

    for n in range(conv_length):
        real = 0.0
        imag = 0.0
        for k in range(conv_length):
            amplitude = y1[k]
            phase = np.pi * 2 * n * k / conv_length
            real += amplitude * np.cos(phase)
            imag += (-amplitude * np.sin(phase))
        complexreal1.append(real)
        compleximag1.append(imag)

    arr, arrr, res = [], [], []
    for i in range(conv_length):
        arr.append(complex(complexreal[i], compleximag[i]))
        arrr.append(complex(complexreal1[i], compleximag1[i]))

    for i in range(conv_length):
        res.append(arr[i] * arrr[i])

    xn = np.fft.ifft(res)
    xn_real = np.real(xn)
    # print("XN_Real")
    # print(xn_real)
    return xn_real

def DCT(y, coefficients):
    y2 = []
    N = len(y)
    for k in range(coefficients):
        dct = 0.0
        for n in range(N):
            dct += np.sqrt(2 / N) * y[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
        y2.append(dct)
    return y2
def Corrolation(y,y1):
    complexreal , compleximag , complexreal1 , compleximag1 ,A , fi , xn ,result_complex_real,result_complex_imag = [],[],[],[],[],[],[],[],[]
    for n in range(len(y)):
        real = 0.0
        imag = 0.0
        for k in range(len(y)):
            amplitude = y[k]
            phase = np.pi*2*n*k /len(y)
            real += amplitude * np.cos(phase)
            imag += (-amplitude * np.sin(phase))
        complexreal.append(real)
        compleximag.append(imag)

    for n in range(len(y1)):
        real = 0.0
        imag = 0.0
        for k in range(len(y1)):
            amplitude = y1[k]
            phase = np.pi*2*n*k /len(y1)
            real += amplitude * np.cos(phase)
            imag += (-amplitude * np.sin(phase))
        complexreal1.append(real)
        compleximag1.append(imag)


    N = len(y)
    arr1 , arr2,res =[], [],[]
    for i in range(N):
        arr1.append(complex(complexreal[i],-1*compleximag[i]))
        arr2.append(complex(complexreal1[i],compleximag1[i]))

    for i in range(N):
        res.append(arr1[i]*arr2[i])

    for i in range(N):
        result_complex_real.append(res[i].real)
        result_complex_imag.append(res[i].imag)

    for i in range(N):
        A.append(np.sqrt(result_complex_real[i]**2 + result_complex_imag[i]**2))
        fi.append(np.arctan2(result_complex_imag[i], result_complex_real[i]))
    # ____________________________IDFT______________________________________
    for n in range(N):
            sum_real = 0.0
            sum_imag = 0.0
            for k in range(len(y)):
                angle = 2 * np.pi * k * n / N
                sum_real += A[k] * np.cos(fi[k] + angle)
                sum_imag += A[k] * np.sin(fi[k] + angle)
            xn.append((1/N)*((sum_real + sum_imag * 1j) / N))
    xn_real = [(np.real(val)) for val in xn]

    return xn_real
def Set_Specifi(FileName):
    global FilterType, FS, StopBandAttenuation, FC, F1, F2, TransitionBand
    FilterType = 0
    FS = 0
    StopBandAttenuation = 0
    FC = 0
    F1 = 0
    F2 = 0
    TransitionBand = 0

    file = open(FileName, "r", encoding='utf-8')
    lines1 = file.readlines()
    file.close()

    x = []
    y = 0

    ignored_lines = lines1[1:]
    for l in ignored_lines:
        row = l.split()
        x.append(float(row[2]))  # Convert to float

    for l in lines1:
        row = l.split()
        y = (row[2] + row[3])
        break

    if y == "Lowpass":
        FilterType = 1
    elif y == "Highpass":
        FilterType = 2
    elif y == "Bandpass":
        FilterType = 3
    elif y == "Bandstop":
        FilterType = 4

    if len(x) == 4:
        FS, StopBandAttenuation, FC, TransitionBand = x
    elif len(x) == 5:
        FS, StopBandAttenuation, F1, F2, TransitionBand = x
    return FilterType, FS, StopBandAttenuation, FC, F1, F2, TransitionBand
def Check_Odd(N):
    if (np.fmod(N, 2) == 1):
        return int(N)
    elif (np.fmod(N, 2) == 0 or (np.fmod(N, 2)) < 1):
        return int(N) + 1
    elif ((np.fmod(N, 2)) > 1):
        return int(N) + 2
def Calculate_FC(Type, TransitionBand, FS, Fc=None, FC1=None, FC2=None):
    if (Type == 1):
        # type = 1 (Low Pass Filter)
        FC_Low_New = (Fc + (TransitionBand / 2)) / FS
        return FC_Low_New
    elif (Type == 2):
        # type = 2 (High Pass Filter)
        FC_High_New = (Fc - (TransitionBand / 2)) / FS
        return FC_High_New
    elif (Type == 3):
        # type = 3 (Band Pass Pass Filter)
        FC1_New = (FC1 - (TransitionBand / 2)) / FS
        FC2_New = (FC2 + (TransitionBand / 2)) / FS
        return FC1_New, FC2_New
    elif (Type == 4):
        # type = 4 (Band Stop Pass Filter)
        FC1_New = (FC1 + (TransitionBand / 2)) / FS
        FC2_New = (FC2 - (TransitionBand / 2)) / FS
        return FC1_New, FC2_New

def Calculate_N(StopBandAttenuation, TransitionBand, FS):
    if (StopBandAttenuation <= 21):
        N = 0.9 * FS / TransitionBand
    elif (StopBandAttenuation > 21 and StopBandAttenuation <= 44):
        N = 3.1 * FS / TransitionBand
    elif (StopBandAttenuation > 44 and StopBandAttenuation <= 53):
        N = 3.3 * FS / TransitionBand
    elif (StopBandAttenuation > 53 and StopBandAttenuation <= 74):
        N = 5.5 * FS / TransitionBand
    N_new = Check_Odd(N)
    return N_new

def Wn(StopBandAttenuation, N, n):
    if (StopBandAttenuation <= 21):
        return 1
    elif (StopBandAttenuation > 21 and StopBandAttenuation <= 44):
        eq = 0.5 + 0.5 * np.cos((2 * np.pi * n) / N)
        return eq
    elif (StopBandAttenuation > 44 and StopBandAttenuation <= 53):
        eq = 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)
        return eq
    elif (StopBandAttenuation > 53 and StopBandAttenuation <= 74):
        eq = 0.42 + 0.5 * np.cos((2 * np.pi * n) / (N - 1)) + 0.08 * np.cos((4 * np.pi * n) / (N - 1))
        return eq
def Calculate_Window(StopBandAttenuation, TransitionBand, FS, Index):
    N = Calculate_N(StopBandAttenuation, TransitionBand, FS)
    n = - int(N / 2)
    wn = []
    for i in range(N):
        wn.append(Wn(StopBandAttenuation, N, n))
        Index.append(n)
        n = n + 1
    return wn

def Calculate_EQUATION(FilterType, n, TransitionBand, FS, FC, F1, F2):
    if (FilterType == 1):
        if (n == 0):
            eq = 2 * Calculate_FC(FilterType, TransitionBand, FS, Fc=FC)
            return eq
        else:
            f = Calculate_FC(FilterType, TransitionBand, FS, Fc=FC)
            eq = 2 * f * np.sin(n * 2 * np.pi * f) / (n * 2 * np.pi * f)
            return eq

    elif (FilterType == 2):
        if (n == 0):
            eq = 1 - 2 * Calculate_FC(FilterType, TransitionBand, FS, Fc=FC)
            return eq
        else:
            f = Calculate_FC(FilterType, TransitionBand, FS, Fc=FC)
            eq = - 2 * f * np.sin(n * 2 * np.pi * f) / (n * 2 * np.pi * f)
            return eq
    elif (FilterType == 3):
        if (n == 0):
            f1, f2 = Calculate_FC(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = 2 * (f2 - f1)
            return eq
        else:
            f1, f2 = Calculate_FC(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = (2 * f2 * np.sin(n * 2 * np.pi * f2) / (n * 2 * np.pi * f2)) - (
                        2 * f1 * np.sin(n * 2 * np.pi * f1) / (n * 2 * np.pi * f1))
            return eq
    elif (FilterType == 4):
        if (n == 0):
            f1, f2 = Calculate_FC(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = 1 - 2 * (f2 - f1)
            return eq
        else:
            f1, f2 = Calculate_FC(FilterType, TransitionBand, FS, FC1=F1, FC2=F2)
            eq = (2 * f1 * np.sin(n * 2 * np.pi * f1) / (n * 2 * np.pi * f1)) - (
                        2 * f2 * np.sin(n * 2 * np.pi * f2) / (n * 2 * np.pi * f2))
            return eq
def Calculate_Filter(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2):
    N = Calculate_N(StopBandAttenuation, TransitionBand, FS)
    n = -int(N / 2)
    hn = []
    for i in range(N):
        hn.append(Calculate_EQUATION(FilterType, n, TransitionBand, FS, FC, F1, F2))
        n = n + 1
    return hn
def Do_FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2):
    Index = []
    wn = Calculate_Window(StopBandAttenuation, TransitionBand, FS, Index)
    hn = Calculate_Filter(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)
    h = []
    N = Calculate_N(StopBandAttenuation, TransitionBand, FS)
    for i in range(N):
        h.append(wn[i] * hn[i])
    return h, Index

def Do_Resampling(m_entry, l_entry):
    file_path = r"resampling\Testcase 1\ecg400.txt"
    file = open(file_path, "r")
    signal_data = file.readlines()
    ignored_lines = signal_data[3:]
    x, y = [], []
    for l in ignored_lines:
        row = l.split()
        x.append(float(row[0]))
        y.append(float(row[1]))

    FilterType, FS, StopBandAttenuation, FC, F1, F2, TransitionBand = Set_Specifi(
        r"resampling\Testcase 1\Filter Specifications.txt")

    LowPass, Index = Do_FIR(FilterType, StopBandAttenuation, TransitionBand, FS, FC, F1, F2)

    M = int(m_entry.get())
    L = int(l_entry.get())

    if M == 0 and L != 0:
        output, outputx, outputy = [], [], []
        y_upsampled, x_upsampled = [], []
        for i in range(len(y) - 1):
            y_upsampled.append(y[i])
            for j in range(L - 1):
                y_upsampled.append(0)
            for j in range(L):
                x_upsampled.append(x[i] + ((x[i + 1] - x[i]) / L) * j)

        y_upsampled.append(y[-1])
        x_upsampled.append(x[-1])

        output = convlution(x_upsampled, y_upsampled, Index, LowPass)
        outputx, outputy = output

        Compare_Signals(r"resampling\Testcase 2\Sampling_Up.txt", outputx, outputy)
    elif M != 0 and L == 0:

        output, outputx, outputy = [], [], []
        output = convlution(x, y, Index, LowPass)
        outputx, outputy = output

        y_downsampled, x_downsampled = [], []
        y_downsampled = outputy[::M]
        for i in range(len(y_downsampled)):
            x_downsampled.append(outputx[i])
        Compare_Signals(r"resampling\Testcase 1\Sampling_Down.txt", x_downsampled, y_downsampled)
    elif M != 0 and L != 0:

        output, outputx, outputy = [], [], []
        y_upsampled, x_upsampled = [], []
        for i in range(len(y) - 1):
            y_upsampled.append(y[i])
            for j in range(L - 1):
                y_upsampled.append(0)
            for j in range(L):
                x_upsampled.append(x[i] + ((x[i + 1] - x[i]) / L) * j)

        y_upsampled.append(y[-1])
        x_upsampled.append(x[-1])

        output = convlution(x_upsampled, y_upsampled, Index, LowPass)
        outputx, outputy = output
        print("outputx = " + str(len(outputx)))
        print("outputy = " + str(len(outputy)))

        y_downsampled, x_downsampled = [], []
        y_downsampled = outputy[::M]
        for i in range(len(y_downsampled)):
            x_downsampled.append(outputx[i])
        print("x_downsampled = " + str(len(x_downsampled)))
        print("y_downsampled = " + str(len(y_downsampled)))
        print(x_downsampled)
        print(y_downsampled)
        Compare_Signals(r"resampling\Testcase 3\Sampling_Up_Down.txt", x_downsampled, y_downsampled)
    else:
        print("Error!!!!!!!")


def Main(ecg_folder_A, ecg_folder_B, test_folder, Fs, miniF, maxF, newFs):
    def filter_signal(signal, Fs, miniF, maxF):
        LowPass0, Index0 = Do_FIR(3, 50, 500, Fs, 0, miniF, maxF)
        outputy = convlution(signal, LowPass0)
        return outputy

    def resample_signal(signal, Fs, newFs):
        if newFs < Fs:
            print("newFs is not valid. Cannot upsample the signal.")
            return signal
        else:
            resample_factor = int(newFs / Fs)
            if resample_factor <= 0:
                print("newFs is not valid. Cannot upsample the signal.")
                return signal
            else:
                fraction = Fraction(resample_factor).limit_denominator()

                L = fraction.numerator
                M = fraction.denominator

                # resampled_signal = resample(signal,M,L)
                resampled_signal = scipy.signal.resample(signal, len(signal) * resample_factor)
                return resampled_signal

    def compute_auto_correlation(signal):
        flattened_signal = signal
        auto_correlation = Corrolation(flattened_signal, flattened_signal)
        return auto_correlation[len(auto_correlation) // 2:]

    def preserve(auto_correlation):
        preserved_coefficients = auto_correlation[:100]
        return preserved_coefficients

    def compute_dct(signal):
        dct_result = DCT(signal, 100)
        return dct_result

    def template_matching(template, test_signal):
        correlation_result = Corrolation(test_signal, template)
        return correlation_result

    def plot_signals(original, auto_corr, preserved_coeff, dct_result, label):
        plt.figure(figsize=(12, 8))

        plt.subplot(511)
        plt.plot(original)
        plt.title('Original Signal')

        plt.subplot(512)
        plt.plot(auto_corr)
        plt.title('Auto-correlation')

        plt.subplot(513)
        plt.plot(preserved_coeff)
        plt.title('Preserved Coefficients of Auto-correlation')

        plt.subplot(514)
        plt.plot(dct_result)
        plt.title('DCT Result')

        plt.subplot(515)
        plt.text(0.5, 0.5, f'Label: {label}', horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.title('Subject Label')

        plt.tight_layout()
        plt.show()

    def load_signal(file_path):

        with open(file_path, 'r') as file:
            signal = [float(line.strip()) for line in file.readlines()]
        return np.array(signal)

    # Replace 'ecg_A_file.txt' and 'ecg_B_file.txt' with the actual file names in your "A" and "B" folders.
    ecg_A = np.mean([load_signal(os.path.join(ecg_folder_A, file)) for file in os.listdir(ecg_folder_A)], axis=0)
    ecg_B = np.mean([load_signal(os.path.join(ecg_folder_B, file)) for file in os.listdir(ecg_folder_B)], axis=0)

    # Filter, resample, remove DC, and normalize signals for subjects A and B
    filtered_A = normalize(
        remove_dc(resample_signal(filter_signal(ecg_A, Fs, miniF, maxF), Fs, newFs)))
    filtered_B = normalize(
        remove_dc(resample_signal(filter_signal(ecg_B, Fs, miniF, maxF), Fs, newFs)))

    # Compute auto-correlation and preserve coefficients for subjects A and B
    auto_corr_A = compute_auto_correlation(filtered_A)
    preserved_coeff_A = preserve(auto_corr_A)

    auto_corr_B = compute_auto_correlation(filtered_B)
    preserved_coeff_B = preserve(auto_corr_B)

    # Compute DCT for subjects A and B
    dct_A = compute_dct(preserved_coeff_A)
    dct_B = compute_dct(preserved_coeff_B)

    # Iterate over test ECG files
    pl = []
    for test_file in os.listdir(test_folder):
        test_signal = np.loadtxt(os.path.join(test_folder, test_file))

        # Filter, resample, remove DC, and normalize test signal
        filtered_test = normalize(
            remove_dc(resample_signal(filter_signal(test_signal, Fs, miniF, maxF), Fs, newFs)))

        # Compute auto-correlation and preserve coefficients for the test signal
        auto_corr_test = compute_auto_correlation(filtered_test)
        preserved= preserve(auto_corr_test)

        # Compute DCT for the test signal
        dct_test = compute_dct(preserved)

        # Use template matching to compare DCT and classify as subject A or B
        correlation_A = template_matching(dct_A, dct_test)
        correlation_B = template_matching(dct_B, dct_test)

        # Classify based on the maximum correlation
        subject_label = 'A' if np.max(correlation_A) > np.max(correlation_B) else 'B'

        # Plot the signals and the classification result
        plot_signals(filtered_test, auto_corr_test, preserved, dct_test, subject_label)



# Example usage:
Main("A", "B", "Test Folder", Fs=1000, miniF=0.5, maxF=50, newFs=2000)