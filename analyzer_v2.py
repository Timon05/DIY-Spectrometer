import cv2
import numpy as np
import matplotlib.pyplot as plt

def pid_control(error, prev_error, integral, Kp=2.0, Ki=0.05, Kd=0.1):
    derivative = error - prev_error
    integral = max(min(integral + error, 1000), -1000)  # Clamping integral
    return Kp * error + Ki * integral + Kd * derivative, derivative, integral

def analyze(image_path, save_path=None):
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Fehler: Bild konnte nicht geladen werden!")
        return None
    
    # Extract color channels
    r_channel = np.mean(frame[:, :, 2], axis=0)
    g_channel = np.mean(frame[:, :, 1], axis=0)
    b_channel = np.mean(frame[:, :, 0], axis=0)
    
    # Compute first and second derivatives
    def calculate_derivative(signal):
        return np.gradient(signal)
    
    def calculate_second_derivative(signal):
        return np.gradient(np.gradient(signal))
    
    first_derivative_r = calculate_derivative(r_channel)
    first_derivative_g = calculate_derivative(g_channel)
    first_derivative_b = calculate_derivative(b_channel)

    second_derivative_r = calculate_second_derivative(r_channel)
    second_derivative_g = calculate_second_derivative(g_channel)
    second_derivative_b = calculate_second_derivative(b_channel)
    
    # Apply moving average to smooth data
    def smooth_signal(signal, window_size=3):
        return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    
    r_channel = smooth_signal(r_channel)
    g_channel = smooth_signal(g_channel)
    b_channel = smooth_signal(b_channel)

    # --- FIND GLOBAL PEAK IN RANGE 600-1000 ---
    peak_range_start = 500
    peak_range_end = 1100
    
    peak_r = np.argmax(r_channel[peak_range_start:peak_range_end]) + peak_range_start
    peak_g = np.argmax(g_channel[peak_range_start:peak_range_end]) + peak_range_start
    peak_b = np.argmax(b_channel[peak_range_start:peak_range_end]) + peak_range_start
    
    global_peak = (peak_r + peak_g + peak_b) // 3
    
    print(f"Global Peak Position: {global_peak}")

    # --- DETECT ANOMALIES OUTSIDE 600-1000 ---
    potential_anomalies = []
    prev_error = 0
    integral = 0
    threshold = np.percentile(np.concatenate([first_derivative_r, first_derivative_g, first_derivative_b]), 98)
    peak_threshold = np.percentile(np.concatenate([second_derivative_r, second_derivative_g, second_derivative_b]), 98)
    
    for i in range(len(r_channel)):
        if peak_range_start <= i <= peak_range_end:  # Ignore the peak range
            continue
        
        dir_r = np.sign(first_derivative_r[i])
        dir_g = np.sign(first_derivative_g[i])
        dir_b = np.sign(first_derivative_b[i])
        #print(first_derivative_b)
        #is_anomaly = not (dir_r == dir_g == dir_b)
        diff = 0.15
        is_anomaly = not ((abs(first_derivative_b[i] - first_derivative_g[i]) < diff)and(abs(first_derivative_r[i] - first_derivative_g[i]) < diff) and (abs(first_derivative_r[i] - first_derivative_b[i]) < diff)  )
        is_peak = max(abs(second_derivative_r[i]), abs(second_derivative_g[i]), abs(second_derivative_b[i])) > peak_threshold
        
        if is_anomaly or is_peak:
            error = max(abs(first_derivative_r[i]), abs(first_derivative_g[i]), abs(first_derivative_b[i]))
            second_derivative_error = max(abs(second_derivative_r[i]), abs(second_derivative_g[i]), abs(second_derivative_b[i]))
            
            correction, derivative, integral = pid_control(error + second_derivative_error, prev_error, integral)
            
            if correction > threshold * 0.8 or is_peak:
                potential_anomalies.append(i)
            
            prev_error = error
    
    # --- GROUP ANOMALIES INTO CLUSTERS ---
    clusters = []
    cluster = []
    cluster_width = 15
    for i in range(len(potential_anomalies)):
        if i == 0 or potential_anomalies[i] - potential_anomalies[i-1] <= 5:
            cluster.append(potential_anomalies[i])
        else:
            if len(cluster) >= cluster_width:
                clusters.append((cluster[0], cluster[-1]))
            cluster = [potential_anomalies[i]]
    if len(cluster) >= cluster_width:
        clusters.append((cluster[0], cluster[-1]))

    # --- PLOT RESULTS ---
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Spectroscopy Image")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.subplot(2, 1, 2)
    plt.xlabel("x-position [pixel]")
    plt.ylabel("Color Intensity")
    plt.plot(r_channel, color='r', label='Red', alpha=0.7)
    plt.plot(g_channel, color='g', label='Green', alpha=0.7)
    plt.plot(b_channel, color='b', label='Blue', alpha=0.7)
    
    plt.axvline(global_peak, color='black', linestyle='dashed', label="Global Peak")
    
    for start, end in clusters:
        plt.axvspan(start, end, color='gray', alpha=0.3, label='Anomaly Region' if start == clusters[0][0] else "")
    
    plt.legend()
    plt.title("Detected Color Anomalies and Peaks with PID Control")
    plt.xlim(0,1600)
    
    if save_path:
        plt.savefig(save_path)
    
    
    return global_peak, clusters

# Example usage:
# global_peak, anomalies = analyze("C:\\Users\\timon\\Downloads\\spektrum_sunlight_1.jpg")
# print(f"Global Peak: {global_peak}")
# print(f"Anomalies: {anomalies}")
