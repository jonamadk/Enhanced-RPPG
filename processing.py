# import the necessary packages
from __future__ import division
from cffi.backend_ctypes import xrange
from scipy.signal.windows import blackmanharris
from common import parabolic as parabolic
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from Kalman_filter2 import *
from numpy.fft import rfft
from numpy import argmax, mean, diff, log


previous = 70
dataset = [0 for i in range(10)]



def openGraph(lock):
    from matplotlib import animation
    def update_line(num, data, line):
        global previous
        # if os._exists('bpmTemp.txt'):
        lock.acquire()
        textFile = open("bpmTemp.txt", 'r')
        bpm = float(textFile.read())
        previous = bpm
        textFile.close()
        lock.release()

        data.append(previous)
        plt.clf()
        plt.plot(data, 'r-')
        x = plt.gca()
        x.set_xlabel('Time (seconds)', fontsize=35)
        x.set_ylabel('B.P.M.', fontsize=35)

        fig1.suptitle('Heart Rate(B.P.M)\n' + str(previous), fontsize=35, fontweight='bold')
        # line.set_ydata(data)
        plt.xlim([num - 9, num + 1])
        return line,

    fig1 = plt.figure(figsize=(10, 10))
    bpmtext = fig1.suptitle('Heart Rate(B.P.M)\n' + str(previous), fontsize=35, fontweight='bold')
    # bpmtext = plt.figtext(0.9, 0.5, 'Heart Rate\n(B.P.M)\n' + str(previous), fontdict=None, \
    #                       fontsize=16, fontweight='bold', position=(1, 0.5))
    # fig1.suptitle('BMP graph', fontsize=14, fontweight='bold')
    x = plt.gca()
    x.set_xlabel('Time (seconds)', fontsize=35)
    x.set_ylabel('B.P.M.', fontsize=35)
    data = [previous for i in xrange(10)]
    l, = plt.plot(data, 'r-')
    plt.ylim(40, 180)

    line_ani = animation.FuncAnimation(fig1, update_line, fargs=(data, l), interval=1000, blit=False)
    plt.show()


# Estimate frequency from peak of FFT
# returns calculated frequency
def freq_from_fft(sig, fs):
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    fe = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(fe))  # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(fe)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


# creates a butter pass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# creates a butter pass filter by cutting off certain frequency
# returns filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian
# retruns variance
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# finds mean of a lsit
# retruns the mean
def findvarmean(alist):
    b = 0
    for a in alist:
        b = b + variance_of_laplacian(a)
    return b / len(alist)


# cheacks particular pixel difference from two lists consisting of pexel location
# returns a boolean depending on the pixel distance
def checkpixeldiff(a, b):
    if b[0] == 0:
        return True
    else:
        c1 = abs(a[0] - b[0])
        c2 = abs(a[1] - b[1])
        c3 = abs(a[2] - b[2])
        c4 = abs(a[3] - b[3])
        if c1 < 50 and c2 < 50 and c3 < 50 and c4 < 50:
            return True
        else:
            return False


# converta frame to HSV
# retruns HSV frame
def CoverttoHSV(image1):
    HSV = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    return HSV


def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

