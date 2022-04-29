import argparse
from concurrent.futures import wait
import cv2 as cv
import numpy as np
from itertools import product
from math import sqrt, cos, pi
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
from dahuffman import HuffmanCodec
import math
import datetime


def extractYUV(file_name, height, width):
    """
    Extracts the Y, U, and V components of the frames in the given video file.
    :param file_name: filepath of video file to extract frames from.
    :param height: height of video.
    :param width: width of video.
    :param start_frame: first frame to be extracted.
    :param end_frame: final frame to be extracted.
    :
    """

    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # Seek to end of file
    fp_end = fp.tell()  # Find the file size

    frame_size = height * width * 3 // 2  # Size of a frame in bytes
    num_frame = fp_end // frame_size  # Number of frames in the video
    print("This yuv file has {} frame imgs!".format(num_frame))
    fp.seek(0, 0)  # Seek to the start of the first frame

    YUV = []
    for i in range(num_frame):
        yuv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yuv[j] = ord(fp.read(1))  # Read one byte from the file

        img = yuv.reshape((height * 3 // 2, width)).astype('uint8')  # Reshape the array    
        
        # YUV420
        y = np.zeros((height, width), dtype='uint8', order='C')
        u = np.zeros((height // 2) * (width // 2), dtype='uint8', order='C')
        v = np.zeros((height // 2) * (width // 2), dtype='uint8', order='C')
        
        # assignment
        y = img[:height, :width]
        u = img[height : height * 5 // 4, :width]
        v = img[height * 5 // 4 : height * 3 // 2, :width]

        # reshape
        u = u.reshape((height // 2, width // 2)).astype('uint8')
        v = v.reshape((height // 2, width // 2)).astype('uint8')

        # save
        YUV.append({'y': y, 'u': u, 'v': v})


    fp.close()
    print("job done!")
    return YUV, num_frame

def YUV2RGB(y, u, v, height, width):
    '''
    Converts YUV to RGB.
    :param y: Y component.
    :param u: U component.
    :param v: V component.
    :param height: height of image.
    :param width: width of image.
    :return: RGB components.
    '''
    yuv = np.zeros((height * 3 // 2, width), dtype='uint8', order='C')
    y = y.reshape((height, width))
    u = u.reshape((-1, width))
    v = v.reshape((-1, width))
    yuv[:height, :width] = y
    yuv[height : height*5//4, :width] = u
    yuv[height*5//4 : height*3//2, :width] = v
    rgb = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420)  
    return rgb

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
    scale = 31

    DC_step_size = 8
    AC_step_size = 2 * scale
    # Perform quantization or its inverse depending on isInv flag.
    if isInv:
        quantized = (mat * AC_step_size).astype(np.int32)
        quantized[0:width:8, 0:height:8] = (mat[0:width:8, 0:height:8] * DC_step_size).astype(np.int32)
    else:
        quantized = (mat / AC_step_size).astype(np.int32)
        quantized[0:width:8, 0:height:8] = (mat[0:width:8, 0:height:8] / DC_step_size).astype(np.int32)
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
    matIdx = np.array([0,  1,  5,  6, 14, 15, 27, 28,
                    2,  4,  7, 13, 16, 26, 29, 42,
                    3,  8, 12, 17, 25, 30, 41, 43,
                    9, 11, 18, 24, 31, 40, 44, 53,
                    10, 19, 23, 32, 39, 45, 52, 54,
                    20, 22, 33, 38, 46, 51, 55, 60,
                    21, 34, 37, 47, 50, 56, 59, 61,
                    35, 36, 48, 49, 57, 58, 62, 63])
    for N, M in product(range(0, height, 8), range(0, width, 8)):
        if N >= height // 8*8 or M >= width // 8*8:
            break
        num = N // 8 * width // 8 + M // 8

        coeffMat[num][matIdx] = mat[N:N+8, M:M+8].reshape(-1)

    return coeffMat

def IextractCoefficients(coeffMat,width,height):
    """
    :Reconstruct block
    :param width: width of frame.
    :param height: height of frame.
    """
    blockMat = np.zeros((height, width))
    matIdx = np.array([0,  1,  5,  6, 14, 15, 27, 28,
                    2,  4,  7, 13, 16, 26, 29, 42,
                    3,  8, 12, 17, 25, 30, 41, 43,
                    9, 11, 18, 24, 31, 40, 44, 53,
                    10, 19, 23, 32, 39, 45, 52, 54,
                    20, 22, 33, 38, 46, 51, 55, 60,
                    21, 34, 37, 47, 50, 56, 59, 61,
                    35, 36, 48, 49, 57, 58, 62, 63])
    for N, M in product(range(0, height, 8), range(0, width, 8)):
        if N >= height // 8*8 or M >= width // 8*8:
            break
        num = N // 8 * width // 8 + M // 8
        blockMat[N:N+8, M:M+8] = coeffMat[num][matIdx].reshape((8,8))

    return blockMat

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
    h_num = math.ceil(height/16)
    w_num = math.ceil(width/16)
    size = h_num*w_num
    MV_arr = MV_subarr = np.zeros((2, size)).astype(int)
    y_pred = np.zeros((height, width))
    cb_pred = cr_pred = np.zeros((height // 2, width // 2))
    coordMat = np.zeros((4,size))
    mv_row = mv_col = 0

    # Search window sizes depend on where the macroblock is in the frame; i.e. at an edge, column/row, or in the middle.
    SW_dict = {
        576: 81,
        768: 153,
        1024: 289
    }

    # For each macroblock in the frame:
    mv_idx = 0
    for n, m in product(range(0, height, 16), range(0, width, 16)):
        MB_curr = y_curr[n:n + 16, m:m + 16]  # Current macroblock.

        # Identify search window parameters. For 8 px in each directions, we can have search windows of sizes 24x24,
        # 24x32, 32x24, or 32x32.
        SW_hmin = 0 if n - 8 < 0 else n - 8
        SW_wmin = 0 if m - 8 < 0 else m - 8
        SW_hmax = height if n + 16 + 8 > height else n + 16 + 8
        SW_wmax = width if m + 16 + 8 > width else m + 16 + 8

        SW_x = SW_wmax - SW_wmin
        SW_y = SW_hmax - SW_hmin
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
        for i, j in product(range(SW_hmin, SW_hmax - 15), range(SW_wmin, SW_wmax - 15)):
            MB_temp = y_ref[i:i + 16, j:j + 16]
            if MB_curr.shape[0] != 16 or MB_curr.shape[1] != 16:
                pass
            else:
                diff = np.float32(MB_curr) - np.float32(MB_temp)
                SAD_vect[SW_idx] = np.sum(np.abs(diff))
                SAD_arr[0, SW_idx] = i
                SAD_arr[1, SW_idx] = j
                SW_idx += 1

        if MB_curr.shape[0] != 16 or MB_curr.shape[1] != 16:
            pass
        else:
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

        if MB_curr.shape[0] != 16 or MB_curr.shape[1] != 16:
            y_pred[n:n + 16, m:m + 16] = np.float32(y_ref[n:n + MB_curr.shape[0], m: m+ MB_curr.shape[1]])
        else:
            # Apply the motion vectors to the current block of the reference frame,
            y_pred[n:n + 16, m:m + 16] = np.float32(y_ref[mv_row:mv_row + 16, mv_col:mv_col + 16])

            # Get motion vector inputs for quiver().
            coordMat[0, mv_idx] = m
            coordMat[1, mv_idx] = n
            coordMat[2, mv_idx] = mv_col - m
            coordMat[3, mv_idx] = mv_row - n

            mv_idx += 1

    # Do the same for cb/cr.
    cbcr_idx = 0
    for i, j in product(range(0, (height // 2), 8), range(0, (width // 2), 8)):
        if i + 8 > (height//2):
            cb_pred[i:height//2,j:j+8] = np.float32(cb_ref[i:height//2,j:j+8])
            cr_pred[i:height//2,j:j+8] = np.float32(cr_ref[i:height//2,j:j+8])
        else:
            ref_row = i + (MV_subarr[0, cbcr_idx])
            ref_col = j + (MV_subarr[1, cbcr_idx])

            cb_pred[i:i + 8, j:j + 8] = np.float32(cb_ref[ref_row:ref_row + 8, ref_col:ref_col + 8])
            cr_pred[i:i + 8, j:j + 8] = np.float32(cr_ref[ref_row:ref_row + 8, ref_col:ref_col + 8])

            cbcr_idx += 1

    return coordMat, MV_arr, MV_subarr, y_pred, cb_pred, cr_pred

def getDC(CoeffMat):
    '''
    Computes DC coefficients for a given YUV component.
    :param CoeffMat: YUV component.
    :return: DC coefficients.
    '''
    dc_coeff = np.zeros(CoeffMat.shape[0])
    dc_coeff = CoeffMat[:, 0]
    dcdpcm = np.zeros(CoeffMat.shape[0])
    dcdpcm[0] = dc_coeff[0]
    dcdpcm[1:] = dc_coeff[1:] - dc_coeff[:-1]
    return dc_coeff, dcdpcm


def getAC(CoeffMat):
    '''
    Computes AC coefficients for a given YUV component using RLE
    :param CoeffMat: YUV component.
    :return: AC coefficients.
    '''
    ac_coeff = []
    for i in range(CoeffMat.shape[0]):
        "using the run length encoding algorithm"
        cnt = 0
        for x in CoeffMat[i, 1:]:
            if x == 0:
                cnt += 1
            if x != 0:
                ac_coeff.append((cnt, x))
                cnt = 0
        ac_coeff.append((0, 0))
    return ac_coeff


def huffmanCoding(data):
    '''
    Huffman coding for data.
    :param data:data.
    :return: Huffman coded and encode.
    '''
    codec = HuffmanCodec.from_data(data)
    encode = codec.encode(data)
    return codec, encode

def MatDecode(dc_codec, dc_encode, ac_codec, ac_encode, num):
    '''
    Decodes DC and AC coefficients.
    :param dc_codec: DC Huffman codec.
    :param dc_encode: DC Huffman encoded coefficients.
    :param ac_codec: AC Huffman codec.
    :param ac_encode: AC Huffman encoded coefficients.
    :return: Decoded DC and AC coefficients.
    '''
    dc_decode = HuffmanCodec.decode(dc_codec, dc_encode)
    dc = np.zeros((num, ))
    dc = dc_decode[:]
    dc = np.cumsum(dc)
    ac_decode = HuffmanCodec.decode(ac_codec, ac_encode)
    Mat = np.zeros((num, 64))
    Mat[:, 0] = dc
    block = 0
    cur = 1
    for ac in ac_decode:
        if ac == (0, 0):
            Mat[block, cur : 64] = 0
            block += 1
            cur = 1
        else:
            cnt = ac[0]
            Mat[block, cur : cur + cnt] = 0
            Mat[block, cur + cnt] = ac[1]
            cur += cnt + 1
        
    return Mat

def encode_decode(y, u, v, height, width):
    '''
    Encodes and decodes the YUV components.
    :param y: YUV component.
    :param u: YUV component.
    :param v: YUV component.
    :param height: Height of the image.
    :param width: Width of the image.
    :return: Encoded and decoded YUV components.
    '''

    yDCT, uDCT, vDCT = dctn(y), dctn(u), dctn(v)

    yQuant = quantize(yDCT, width, height)
    uQuant = quantize(uDCT, width // 2, height // 2)
    vQuant = quantize(vDCT, width // 2, height // 2)

    # Extract DC and AC coefficients; these would be transmitted to the decoder in a real MPEG
    # encoder/decoder framework.
    yCoeffMat = extractCoefficients(yQuant, width, height)
    
    dc_y, dpcm_y = getDC(yCoeffMat)
    ac_y = getAC(yCoeffMat)
    dccodec_y, dcencode_y = huffmanCoding(dpcm_y)
    accodec_y, acencode_y = huffmanCoding(ac_y)
    
    uCoeffMat = extractCoefficients(uQuant, width // 2, height // 2)
    dc_u, dpcm_u = getDC(uCoeffMat)
    ac_u = getAC(uCoeffMat)
    dccodec_u, dcencode_u = huffmanCoding(dpcm_u)
    accodec_u, acencode_u = huffmanCoding(ac_u)

    vCoeffMat = extractCoefficients(vQuant, width // 2, height // 2)
    dc_v, dpcm_v = getDC(vCoeffMat)
    ac_v = getAC(vCoeffMat)
    dccodec_v, dcencode_v = huffmanCoding(dpcm_v)
    accodec_v, acencode_v= huffmanCoding(ac_v)

    # Perform inverse quantization.
    # decoding
    YMatRecon = MatDecode(dccodec_y, dcencode_y, accodec_y, acencode_y, yCoeffMat.shape[0])
    YQuantRecon = IextractCoefficients(YMatRecon, width, height)

    vMatRecon = MatDecode(dccodec_v,dcencode_v,accodec_v,acencode_v,vCoeffMat.shape[0])
    vQuantRecon = IextractCoefficients(vMatRecon,width//2,height//2)

    uMatRecon = MatDecode(dccodec_u,dcencode_u,accodec_u,acencode_u,uCoeffMat.shape[0])
    uQuantRecon = IextractCoefficients(uMatRecon,width//2,height//2)
    
    # perform inverse quantization
    yIQuant = quantize(YQuantRecon, width, height, isInv=True)
    uIQuant = quantize(uQuantRecon, width // 2, height // 2, isInv=True, isLum=False)
    vIQuant = quantize(vQuantRecon, width // 2, height // 2, isInv=True, isLum=False)

    #perform inverse DCT
    yIDCT = idctn(yIQuant)
    uIDCT = idctn(uIQuant)
    vIDCT = idctn(vIQuant)
    
    return yIDCT, uIDCT, vIDCT



def main():
    #desc = 'Showcase of image processing techniques in MPEG encoder/decoder framework.'
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', dest='src', required=True)
    parser.add_argument('--size', dest='size', required=True)
    parser.add_argument('--fps', dest='fps', required=True)
    parser.add_argument('--dst', dest='dst', required=True)

    args = parser.parse_args()

    # Get arguments
    filepath = args.src
    width, height = map(int, args.size.split('x'))
    fps = int(args.fps)
    start_frame = 0
    end_frame = 150
    dst = args.dst
    # print start time
    print('Start time: ' + str(datetime.datetime.now()))
    frames, num_frame = extractYUV(filepath, height, width)
    print('End time: ' + str(datetime.datetime.now()))

    video = cv.VideoWriter(dst, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    i = 0
    PSNR = []
    for frame_num in range(len(frames)):
        curr = frames[frame_num]
        if curr is None:
            continue
        yCurr, uCurr, vCurr = curr['y'], curr['u'], curr['v']
        if frame_num % 2 == 0:
            print("[I] compressing frame " + str(frame_num))

            yRcn, uRcn, vRcn = encode_decode(yCurr, uCurr, vCurr, height, width)

            re_rgb = YUV2RGB(yRcn.astype(np.uint8),uRcn.astype(np.uint8), vRcn.astype(np.uint8), height, width)

        else:
            print("[P] compressing frame " + str(frame_num))

            

            # Do motion estimatation using the I-frame as the reference frame for the current frame in the loop.python mpeg.py --file 'walk_qcif.avi' --extract 6 10
            coordMat, MV_arr, MV_subarr, yPred, uPred, vPred = motionEstimation(yCurr, yCurr, uCurr, vCurr, width,height)

            yTmp, uTmp, vTmp = yPred, uPred, vPred

            # Get residual frame
            yDiff = yCurr.astype(np.uint8) - yTmp.astype(np.uint8)
            uDiff = uCurr.astype(np.uint8) - uTmp.astype(np.uint8)
            vDiff = vCurr.astype(np.uint8) - vTmp.astype(np.uint8)

            yIDCT, uIDCT, vIDCT = encode_decode(yDiff, uDiff, vDiff, height, width)

            yRcn = yIDCT.astype(np.uint8) + yPred.astype(np.uint8)
            uRcn = uIDCT.astype(np.uint8) + uPred.astype(np.uint8)
            vRcn = vIDCT.astype(np.uint8) + vPred.astype(np.uint8)

            i += 1
            re_rgb = YUV2RGB(yRcn.astype(np.uint8),uRcn.astype(np.uint8),vRcn.astype(np.uint8), height, width)
            diffMat = YUV2RGB(yDiff, uDiff, vDiff, height, width)
            pred_rgb = YUV2RGB(yTmp, uTmp, vTmp, height, width)
            plt.figure(figsize=(10, 10))
            curr = YUV2RGB(yCurr, uCurr, vCurr, height, width)
            curr_plt = cv.cvtColor(curr, cv.COLOR_BGR2RGB)
            re_rgb_plt = cv.cvtColor(re_rgb, cv.COLOR_BGR2RGB)
            pred_rgb_plt = cv.cvtColor(pred_rgb, cv.COLOR_BGR2RGB)
            diffMat_plt = cv.cvtColor(diffMat, cv.COLOR_BGR2RGB)
            plt.subplot(2, 2, 1).set_title('Current Image'), plt.imshow(curr_plt)
            plt.subplot(2, 2, 3).set_title('Differential Image'), plt.imshow(diffMat_plt)
            plt.subplot(2, 2, 2).set_title('Predicted Image'), plt.imshow(pred_rgb_plt)
            plt.subplot(2, 2, 4).set_title('Motion Vectors'), plt.quiver(coordMat[0, :], coordMat[1, :], coordMat[2, :],
                                                                        coordMat[3, :])
            plt.savefig('result/train_'+str(i)+'.png')     
            plt.close()
            
        mse = (np.linalg.norm(yRcn - yCurr) + np.linalg.norm(uRcn - uCurr)  + np.linalg.norm(vRcn - vCurr)) \
                / (height * width * 3 // 2)
        psnr = 10 * np.log10(255 * 255 / mse)
        PSNR.append(psnr)

        video.write(re_rgb)  
    plt.title('PSNR per Frame')
    plt.ylim([50, 100])
    plt.plot(PSNR)
    plt.show()


if __name__ == '__main__':
    main()