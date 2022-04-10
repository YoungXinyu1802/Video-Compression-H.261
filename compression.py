import argparse
import cv2 as cv
import numpy as np
from itertools import product
from math import sqrt, cos, pi
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt


# Options for debugging.
# np.set_printoptions(threshold=np.inf)


def extractFrames(video, f1, f2):
    '''
    Extracts a number of BGR frames specifically from the f1-th to the f2-th frame of the given video file.
    :param video: filepath of video file to extract frames from.
    :param f1: the first frame to be extracted.
    :param f2: the final frame to be extracted.
    :return: a list of the extracted BGR frames, as well as with the pixel dimensions of the video.
    '''
    # Capture video and initialize list.
    cap = cv.VideoCapture(video)
    frames = []

    # Get auxiliary information about video.
    width = int(cap.get(3))
    height = int(cap.get(4))
    nframes = int(cap.get(7))
    ret = True

    # Until we reach the end of the sequence, read each frame and append to list.
    while ret:
        ret, frame = cap.read()
        frames.append(frame)

    # Extract the specific frames dependant on the input.
    extracted_frames = [frames[idx] for idx in range(f1 - 1, f2)]
    print(f'Read {nframes} frames from input video file, extracting only {f2 - f1 + 1} frames.')

    return extracted_frames, width, height


def BGR2YCRCB(image):
    '''
    Converts an image from BGR to YCrCb format.
    :param image: input image to be converted.
    :return: split y, cr, and cb components of the converted YCrCb image.
    '''
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)

    return y, cr, cb


def subsample_420(cb, cr, width, height):
    '''
    Subsamples the Cb and Cr component images to 4:2:0 format.
    :param cb: Cb component of YCrCb image.
    :param cr: Cr component of YCrCb image.
    :param width: width of image.
    :param height:height of image.
    :return: 4:2:0 subsampled Cb and Cr components.
    '''
    cb_sub = cb[0:height:2, 0:width:2]
    cr_sub = cr[0:height:2, 0:width:2]

    return cb_sub, cr_sub


def linearInterpolate(mat, width, height):
    '''
    Upsamples an image from 4:2:0 format to be the size of the original image using linear interpolation.
    :param mat: input image matrix.
    :param width: width of matrix.
    :param height: height of matrix.
    :return: linearly interpolated upsampled image.
    '''
    linMat = np.zeros((height, width))
    # Fill linMat with subsampled values with 'empty' value between each subsampled value to allow for averaging later.
    linMat[0:height:2, 0:width:2] = mat[0:height // 2, 0:width // 2]

    # This fills the 'empty' pixel values in odd rows.
    for cols, rows in product(range(1, width, 2), range(0, height, 2)):
        # If an edge is reached, fill with previous value. Otherwise take the average of the neighbors.
        if cols == width - 1:
            linMat[rows, cols] = linMat[rows, cols - 1]
        else:
            linMat[rows, cols] = (linMat[rows, cols - 1]) / 2 + (linMat[rows, cols + 1]) / 2

    # This fills the 'empty' pixel values in even rows.
    for cols, rows in product(range(width), range(1, height, 2)):
        if rows == height - 1:
            linMat[rows, cols] = linMat[rows - 1, cols]
        else:
            linMat[rows, cols] = (linMat[rows - 1, cols]) / 2 + (linMat[rows + 1, cols]) / 2

    return linMat


def YCRCB2RGB(y, cr, cb, width, height, upTrue=True):
    '''
    Converts an image from YCrCb to RGB format.
    :param y: Y component.
    :param cr: Cr component.
    :param cb: Cb component.
    :param width: width of image.
    :param height: height of image.
    :param upTrue: flag indicating whether the image needs to be upsampled from 4:2:0 format.
    :return: reconstructed RGB image.
    '''
    # Upsample image if needed.
    if upTrue:
        cr = linearInterpolate(cr, width, height).astype(np.uint8)
        cb = linearInterpolate(cb, width, height).astype(np.uint8)

    y = y.astype(np.uint8)
    cr = cr.astype(np.uint8)
    cb = cb.astype(np.uint8)

    reconstructed_ycrcb = cv.merge((y, cr, cb))
    reconstructed_rgb = cv.cvtColor(reconstructed_ycrcb, cv.COLOR_YCrCb2RGB)

    return reconstructed_rgb


# NOT CURRENTLY USED
def DCT(mat, width, height):
    '''
    Computes 2-D discrete cosine transform on an input image.
    :param mat: input image matrix.
    :param width: width of image.
    :param height: height of image.
    :return: DCT coefficient matrix.
    '''
    coeffMat = np.zeros((height + 4, width)) if height == 264 else np.zeros((height, width))
    cm = cn = 1.0
    sum = 0.0

    # Loop through N, M to define 8x8 blocks; cols, rows defines each coefficient value within the block.
    for N, M in product(range(0, width, 8), range(0, height, 8)):
        for cols, rows in product(range(8), repeat=2):
            cm = (1.0 / sqrt(2.0)) if rows == 0 else 1.0
            cn = (1.0 / sqrt(2.0)) if cols == 0 else 1.0
            # Compute the DCT sum.
            for j, i in product(range(8), repeat=2):
                sum += float(mat[M + i - 1, N + j - 1]) * cos(((2.0 * i) * (rows - 1) * pi) / 16.0) * cos(
                    ((2.0 * j) * (cols - 1) * pi) / 16.0)
            # Using the sum, cn, cm values compute the coefficient value.
            coeffMat[M + rows - 1, N + cols - 1] = (1.0 / 4.0) * cn * cm * sum
            sum = 0.0

    return coeffMat


# NOT CURRENTLY USED
def IDCT(mat, width, height):
    '''
    Computes 2-D inverse discrete cosine transform on an input image.
    :param mat: input image matrix
    :param width: width of image.
    :param height: height of image.
    :return: matrix representing original pixel values.
    '''
    coeffMat = np.zeros((height + 4, width)) if height == 264 else np.zeros((height, width))
    cm = cn = 1.0
    sum = 0.0

    # Loop through N, M to define 8x8 blocks; cols, rows defines each coefficient value within the block.
    for N, M in product(range(0, width, 8), range(0, height, 8)):
        for j, i in product(range(8), repeat=2):
            for cols, rows in product(range(8), repeat=2):
                cm = (1.0 / sqrt(2.0)) if rows == 0 else 1.0
                cn = (1.0 / sqrt(2.0)) if cols == 0 else 1.0

                sum += cn * cm * float(mat[M + rows - 1, N + cols - 1]) * cos(
                    ((2.0 * i) * (rows - 1) * pi) / 16.0) * cos(
                    ((2.0 * j) * (cols - 1) * pi) / 16.0)

            coeffMat[M + i - 1, N + j - 1] = round((1.0 / 4.0) * sum)
            sum = 0.0

    return coeffMat


def quantize(mat, width, height, isInv=False, isLum=True):
    '''
    Performs quantization or its inverse operation on an image matrix.
    :param mat: DCT coefficient matrix or quantized image matrix.
    :param width: width of matrix.
    :param height: height of matrix.
    :param isInv: flag indicating whether inverse quantization is to be performed.
    :param isLum: flag indicating which image quantization matrix should be used (luminance for Y component, chrominance for Cb/Cr components.).
    :return: image matrix that has undergone quantization or its inverse.
    '''
    quantized = np.zeros((height, width))

    # Luminance quantization matrix (for Y component).
    LQ = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 89, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 108, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

    # Chrominance quantization matrix (for Cb/Cr components).
    CQ = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]])

    # Choose quantization matrix based on isLum flag.
    quantMat = LQ if isLum else CQ

    # Perform quantization or its inverse depending on isInv flag.
    if isInv:
        for N, M, cols, rows in product(range(0, width, 8), range(0, height, 8), range(8), range(8)):
            # for cols, rows in product(range(8), repeat=2):
            quantized[M + rows - 1, N + cols - 1] = round(mat[M + rows - 1, N + cols - 1] * quantMat[rows, cols])
    else:
        for N, M, cols, rows in product(range(0, width, 8), range(0, height, 8), range(8), range(8)):
            # for cols, rows in product(range(8), repeat=2):
            quantized[M + rows - 1, N + cols - 1] = round(mat[M + rows - 1, N + cols - 1] / quantMat[rows, cols])

    return quantized


def extractCoefficients(mat, width, height):
    '''
    Extracts the DC and AC coefficients of the quantized 8x8 block within a frame and places it in a single row of a
    coefficient matrix according to zigzag pattern.
    :param mat: input image matrix.
    :param width: width of image.
    :param height: height of image.
    :return: coefficent matrix with 64 DC and AC coefficents for column values, for each pixel of the 8x8 block.
    '''
    numRows = (height // 8) * (width // 8)  # No. of rows in coefficient matrix is number of 8x8 blocks in the image.
    coeffMat = np.zeros((numRows, 64))
    matIdx = 0

    for N, M in product(range(0, height, 8), range(0, width, 8)):
        coeffBlock = np.zeros(64, )  # Vector representing a single row of the coefficient matrix.
        prevCoord = np.zeros((2, 64))  # Matrix used to store the coordinates of the previous square in the 8x8
        # block according the zigzag pattern.

        # Start from top left of 8x8 block.
        rows = N + 1
        cols = M + 1

        for i in range(64):
            coeffBlock[i] = mat[rows, cols]  # Save current coordinate in the vector.

            # Save current coordinates for next loop.
            prevCoord[0, i] = rows
            prevCoord[1, i] = cols

            # TOP LEFT: move to the right.
            if rows == N & cols == M:
                cols += 1
            # TOP RIGHT: move down and to the left.
            elif rows == N & cols == M + 7:
                rows += 1
                cols -= 1
            # BOTTOM LEFT: move to the right.
            elif rows == N + 7 & cols == M:
                cols += 1
            # ALONG TOP ROW: if previous position along top row, move down and to the left. Else, move to the right.
            elif rows == N:
                if prevCoord[0, i - 1] == N:
                    rows += 1
                    cols -= 1
                else:
                    cols += 1
            # ALONG BOTTOM ROW: if previous position along bottom row, move up and to the right. Else, move to the
            # right.
            elif rows == N + 7:
                if prevCoord[0, i - 1] == N + 7:
                    rows -= 1
                    cols += 1
                else:
                    cols += 1
            # ALONG LEFTMOST COLUMN: if previous position along leftmost column, move up and to the right. Else,
            # move down.
            elif cols == M:
                if prevCoord[1, i - 1] == M:
                    rows -= 1
                    cols += 1
                else:
                    rows += 1
            # ALONG RIGHTMOST COLUMN: if previously along rightmost column, move down and to the left. Else, move down.
            elif cols == M + 7:
                if prevCoord[1, i - 1] == M + 7:
                    rows += 1
                    cols -= 1
                else:
                    rows += 1
            # ELSE: if previous position was one row down, we are moving up and right. Else, we are moving down and
            # left.
            elif (cols > M | cols < M + 7) & (rows > N | rows < N + 7):
                if prevCoord[0, i - 1] == rows + 1:
                    rows -= 1
                    cols += 1
                else:
                    rows += 1
                    cols -= 1
            elif rows == N + 7 & cols == M + 7:
                break

        # Save coefficients for one 8x8 block as a row in the coefficent matrix.
        coeffVect = np.zeros(64)
        coeffVect[:] = coeffBlock
        coeffMat[matIdx, :] = coeffVect
        matIdx += 1

    return coeffMat


def motionEstimation(y_curr, y_ref, cr_ref, cb_ref, width, height):
    '''
    Computes motion estimation for an image based on its reference frame.
    :param y_curr: Y component of current frame; motion estimation is soley done on Y component.
    :param y_ref: Y component of reference frame.
    :param cr_ref: Cr component of reference frame.
    :param cb_ref: Cb component of reference frame.
    :param width: width of frame.
    :param height: height of frame.
    :return: YCrCb components of predicted frame, coordinate matrix for quiver plot, and motion vector matrices.
    '''
    MV_arr = MV_subarr = np.zeros((2, 99)).astype(int)
    y_pred = np.zeros((height, width))
    cb_pred = cr_pred = np.zeros((height // 2, width // 2))
    coordMat = np.zeros((4, 99))
    mv_row = mv_col = 0

    # Search window sizes depend on where the macroblock is in the frame; i.e. at an edge, column/row, or in the middle.
    SW_dict = {
        576: 81,
        768: 153,
        1024: 289
    }

    # For each macroblock in the frame:
    mv_idx = 0
    for n, m in product(range(0, height - 16, 16), range(0, width - 16, 16)):
        MB_curr = y_curr[n:n + 15, m:m + 15]  # Current macroblock.

        # Identify search window parameters. For 8 px in each directions, we can have search windows of sizes 24x24,
        # 24x32, 32x24, or 32x32.
        SW_hmin = 0 if n - 8 < 0 else n - 8
        SW_wmin = 0 if m - 8 < 0 else m - 8
        SW_hmax = height if n + 16 - 1 + 8 > height else n + 16 - 1 + 8
        SW_wmax = width if m + 16 - 1 + 8 > width else m + 16 - 1 + 8

        SW_x = SW_wmax - SW_wmin + 1
        SW_y = SW_hmax - SW_hmin + 1
        SW_size = int(SW_x * SW_y)

        # No. of candidate blocks == search window area.
        SAD_len = 0
        for x, y in SW_dict.items():
            if x == SW_size:
                SAD_len = y
                break
        SAD_vect = np.zeros(SAD_len)
        SAD_arr = np.zeros((2, SAD_len)).astype(int)
        for i in range(SAD_len):
            SAD_vect[i] = 99999.0
            SAD_arr[0, i] = -1
            SAD_arr[1, i] = -1

        # Go through the designated search window for the current macroblock.
        SW_idx = 0
        for i, j in product(range(SW_hmin, SW_hmax - 16), range(SW_wmin, SW_wmax - 16)):
            MB_temp = y_ref[i:i + 15, j:j + 15]
            diff = np.float32(MB_curr) - np.float32(MB_temp)

            SAD_vect[SW_idx] = np.sum(np.abs(diff))
            SAD_arr[0, SW_idx] = i
            SAD_arr[1, SW_idx] = j
            SW_idx += 1

        # Get minimum SAD (sum of absolute differences) and search for its corresponding coordinates.
        SAD_min = min(SAD_vect)
        for i in range(SAD_len):
            if SAD_vect[i] == SAD_min:
                mv_row = (SAD_arr[0, i])
                mv_col = (SAD_arr[1, i])
                break

        # The coordinates gives the the top left pixel + the motion vector coordinates dx and dy.
        MV_arr[0, mv_idx] = mv_row - n
        MV_arr[1, mv_idx] = mv_col - m

        # Do the same for cb/cr, which are subsampled.
        MV_subarr[0, mv_idx] = int((mv_row - n) // 2)
        MV_subarr[1, mv_idx] = int((mv_col - m) // 2)

        # Apply the motion vectors to the current block of the reference frame,
        y_pred[n:n + 15, m:m + 15] = np.float32(y_ref[mv_row:mv_row + 15, mv_col:mv_col + 15])

        # Get motion vector inputs for quiver().
        coordMat[0, mv_idx] = m
        coordMat[1, mv_idx] = n
        coordMat[2, mv_idx] = mv_col - m
        coordMat[3, mv_idx] = mv_row - n

        mv_idx += 1

    # Do the same for cb/cr.
    cbcr_idx = 0
    for i, j in product(range(0, (height // 2) - 8, 8), range(0, (width // 2) - 8, 8)):
        ref_row = i + (MV_subarr[0, cbcr_idx])
        ref_col = j + (MV_subarr[1, cbcr_idx])

        cb_pred[i:i + 7, j:j + 7] = np.float32(cb_ref[ref_row:ref_row + 7, ref_col:ref_col + 7])
        cr_pred[i:i + 7, j:j + 7] = np.float32(cr_ref[ref_row:ref_row + 7, ref_col:ref_col + 7])

        cbcr_idx += 1

    return coordMat, MV_arr, MV_subarr, y_pred, cb_pred, cr_pred


def main():
    desc = 'Showcase of image processing techniques in MPEG encoder/decoder framework.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--file', dest='filepath', required=True)
    parser.add_argument('--extract', nargs='+', type=int)
    args = parser.parse_args()

    # Get arguments
    filepath = args.filepath
    (f1, f2) = tuple(args.extract)

    frames, width, height = extractFrames(filepath, f1, f2)
    num = f2 - f1 + 1

    for frame_num in range(1, num):
        ref = frames[frame_num - 1]
        curr = frames[frame_num]

        img_rgb = cv.cvtColor(frames[frame_num], cv.COLOR_BGR2RGB)

        yRef, crRef, cbRef = BGR2YCRCB(ref)
        yCurr, crCurr, cbCurr = BGR2YCRCB(curr)

        yIFrame = np.zeros((height, width))
        cbIFrame = np.zeros((height // 2, width // 2))
        crIFrame = np.zeros((height // 2, width // 2))

        cbRef, crRef = subsample_420(cbRef, crRef, width, height)
        cbCurr, crCurr = subsample_420(cbCurr, crCurr, width, height)

        ###########
        # ENCODER #
        ###########
        if frame_num == 1:
            iframeOrig = YCRCB2RGB(yRef, crRef, cbRef, width, height)

            # Perform DCT.
            yDCT = dctn(yRef)
            cbDCT = dctn(cbRef)
            crDCT = dctn(crRef)

            # Perform quantization.
            yQuant = quantize(yDCT, width, height)
            cbQuant = quantize(cbDCT, width // 2, height // 2, isLum=False)
            crQuant = quantize(crDCT, width // 2, height // 2, isLum=False)

            # Extract DC and AC coefficients; these would be transmitted to the decoder in a real MPEG
            # encoder/decoder framework.
            yCoeffMat = extractCoefficients(yQuant, width, height)
            cbCoeffMat = extractCoefficients(cbQuant, width // 2, height // 2)
            crCoeffMat = extractCoefficients(crQuant, width // 2, height // 2)

            # Perform inverse quantization.
            yIQuant = quantize(yQuant, width, height, isInv=True)
            cbIQuant = quantize(cbQuant, width // 2, height // 2, isInv=True, isLum=False)
            crIQuant = quantize(crQuant, width // 2, height // 2, isInv=True, isLum=False)

            # Perform inverse DCT.
            yIDCT = idctn(yIQuant)
            cbIDCT = idctn(cbIQuant)
            crIDCT = idctn(crIQuant)

            # Our only reference frame for the GOP is the I-frame.
            yIFrame = yIDCT
            cbIFrame = cbIDCT
            crIFrame = crIDCT

            # For first loop, pass I-frame to decoder and reconstruct.
            iframeRecon = YCRCB2RGB(yIFrame.astype(np.uint8), crIFrame.astype(np.uint8), cbIFrame.astype(np.uint8),
                                    width, height)

            # Display the original I-frame and its reconstruction at the decoder.
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1).set_title('Original I-frame'), plt.imshow(iframeOrig)
            plt.subplot(1, 2, 2).set_title('Reconstructed I-frame'), plt.imshow(iframeRecon)
            plt.show()

        # Do motion estimatation using the I-frame as the reference frame for the current frame in the loop.python mpeg.py --file 'walk_qcif.avi' --extract 6 10
        coordMat, MV_arr, MV_subarr, yPred, cbPred, crPred = motionEstimation(yCurr, yIFrame, crIFrame, cbIFrame, width,
                                                                              height)

        yTmp = yPred
        cbTmp = cbPred
        crTmp = crPred

        # Get residual frame
        yDiff = yCurr.astype(np.uint16) - yTmp.astype(np.uint16)
        cbDiff = cbCurr.astype(np.uint16) - cbTmp.astype(np.uint16)
        crDiff = crCurr.astype(np.uint16) - crTmp.astype(np.uint16)

        # Perform DCT.
        yDCT = dctn(yDiff)
        cbDCT = dctn(cbDiff)
        crDCT = dctn(crDiff)

        # Perform quantization.
        yQuant = quantize(yDCT, width, height)
        cbQuant = quantize(cbDCT, width // 2, height // 2, isLum=False)
        crQuant = quantize(crDCT, width // 2, height // 2, isLum=False)

        # Extract DC and AC coefficients; these would be transmitted to the decoder in a real MPEG encoder/decoder
        # framework.
        yCoeffMat = extractCoefficients(yQuant, width, height)
        yCoeffMat = extractCoefficients(cbQuant, width // 2, height // 2)
        yCoeffMat = extractCoefficients(crQuant, width // 2, height // 2)

        ###########
        # DECODER #
        ###########
        # Perform inverse quantization.
        yIQuant = quantize(yQuant, width, height, isInv=True)
        cbIQuant = quantize(cbQuant, width // 2, height // 2, isInv=True, isLum=False)
        crIQuant = quantize(crQuant, width // 2, height // 2, isInv=True, isLum=False)

        # Perform inverse DCT.
        yIDCT = idctn(yIQuant)
        cbIDCT = idctn(cbIQuant)
        crIDCT = idctn(crIQuant)

        # Our only reference frame for the GOP is the I-frame.
        yRcn = yIDCT.astype(np.uint8) + yPred.astype(np.uint8)
        cbRcn = cbIDCT.astype(np.uint8) + cbPred.astype(np.uint8)
        crRcn = crIDCT.astype(np.uint8) + crPred.astype(np.uint8)

        # Merge frames and convert to RGB.
        rgb_recon = YCRCB2RGB(yRcn, crRcn, cbRcn, width, height)

        # Prepare residual frame so it can be shown with imshow().
        diffMin = np.amin(yDiff)
        diffMat = yDiff - diffMin
        for i, j in product(range(height), range(width)):
            if diffMat[i, j] > 255:
                diffMat[i, j] = 255
        diffMat = diffMat.astype(np.uint8)

        # For each P-frame in the GOP, display original, reconstructed, & residual images and the quiver plot of the
        # motion vectors used frame-to-frame.
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1).set_title('Original Image'), plt.imshow(img_rgb)
        plt.subplot(2, 2, 2).set_title('Reconstructed Image'), plt.imshow(rgb_recon)
        plt.subplot(2, 2, 3).set_title('Residual Image'), plt.imshow(diffMat)
        plt.subplot(2, 2, 4).set_title('Motion Vectors'), plt.quiver(coordMat[0, :], coordMat[1, :], coordMat[2, :],
                                                                     coordMat[3, :])
        plt.show()


if __name__ == '__main__':
    main()
