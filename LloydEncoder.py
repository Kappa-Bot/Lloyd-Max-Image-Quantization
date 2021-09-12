import numpy as np
import imageio

from math import floor, ceil

import sys
from os.path import isfile
import struct

def main(argv):
    if len(argv) != 2:
        print("Usage: cat_quantizer.py <input_file> <output_file>")
        sys.exit(1)
    inCat, outCat = argv[0], argv[1]
    if not isfile(inCat):
        print("{} is not a valid file".format(inCat))
        sys.exit(2)
    print("Going to encode {} to output file {}".format(inCat, outCat))
    ###################################################################
    maxIters = 20
    nBits = 3; nIntervals = 2 ** nBits
    cat = np.array(imageio.imread(inCat), dtype=np.float32); qCat = np.copy(cat)
    imWidth, imHeight, imBands = cat.shape[0], cat.shape[1], cat.shape[2]
    imPixelsFactor = 1 / (imWidth * imHeight)

    catMin = np.zeros([imBands], dtype=np.uint8); catMax = np.copy(catMin)
    for i in range(imWidth):
        for j in range(imHeight):
            for k in range(imBands):
                catMin[k] = catMin[k] if catMin[k] <= cat[i][j][k] else cat[i][j][k]
                catMax[k] = catMax[k] if catMax[k] >= cat[i][j][k] else cat[i][j][k]
    print("Range of values\t[{}-{}]\t[{}-{}]\t[{}-{}]"
          .format(catMin[0], catMax[0],
                  catMin[1], catMax[1],
                  catMin[2], catMax[2]))

    MSE = np.zeros([imBands], dtype=np.float32)

    # Initially  mid points of intervals: [15.9375, 47.8125, 79.6825, 111.5625, 143.4375, 175.3125, 207.1875, 239.0625]
    rep = np.array([[l * (catMax[k] - catMin[k]) / nIntervals + (catMax[k] - catMin[k]) / (2 * nIntervals)
                     for l in range(nIntervals)] for k in range(imBands)], dtype=np.float32)
    # Void because why not mate, kinda cool tho
    tre = np.zeros([imBands, nIntervals + 1], dtype=np.float32)
    tre[:][0] = 0       # Equivalent to -np.inf
    tre[:][-1] = 255    # Equivalent to  np.inf

    # Estimate Probability Density Function (PDF) of pixels on the original image
    catInt = np.array(imageio.imread(inCat), dtype=np.uint8)
    prb = np.zeros([imBands, 256], dtype=np.uint8)
    for i in range(imWidth):
        for j in range(imHeight):
            for k in range(imBands):
                prb[k][int(catInt[i][j][k])] += 1
    prb = np.divide(prb, imWidth * imHeight)
    ###################################################################
    # Iterate
    for iter in range(maxIters):
        print("ITERATION {}".format(iter), end='', flush=True)

        # Recalculate the Intervals for new iteration
        # Middle point between Representation values
        for k in range(imBands):
            for l in range(1, nIntervals):
                #  r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7
                # 1 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | 255
                tre[k][l] = (rep[k][l - 1] + rep[k][l]) / 2

        print(".", end='', flush=True)

        # Calculate the new representation values
        for k in range(imBands):
            for l in range(nIntervals):
                numerator = 0; denominator = 0
                # 1 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | 255
                #  r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7
                for j in range(floor(tre[k][l]), ceil(tre[k][l + 1]) + 1):
                    numerator += (j * prb[k][j])
                    denominator += prb[k][j]
                if denominator != 0:
                    rep[k][l] = numerator / denominator

        print(".", end='', flush=True)

        # Quantization: replace values with a representative one
        # according to the interval the value is located into
        qCat = np.copy(cat)
        # Calculate MSE and distortion
        newMSE = np.zeros(MSE.shape, dtype=np.float32)
        for i in range(imWidth):
            for j in range(imHeight):
                for k in range(imBands):
                    verif = 0
                    for l in np.array(range(nIntervals), dtype=np.uint8):
                        if cat[i][j][k] > tre[k][l] \
                                and cat[i][j][k] <= tre[k][l + 1]:
                            qCat[i][j][k] = rep[k][l]
                            verif = 1
                    if verif == 0:
                        if cat[i][j][k] > - 1 and cat[i][j][k] < tre[k][0]:
                            qCat[i][j][k] = rep[k][0]
                        elif cat[i][j][k] < 256 and cat[i][j][k] > tre[k][nIntervals - 1]:
                            qCat[i][j][k] = rep[k][nIntervals - 1]
                        else:
                            qCat[i][j][k] = rep[k][0]
                    newMSE[k] += ((qCat[i][j][k] - cat[i][j][k]) ** 2)
        newMSE = np.array([mse / (imWidth * imHeight) for mse in newMSE], dtype=np.float64)

        print(".", end='', flush=True)

        if iter > 0:
            distortion = np.array(np.round([(MSE[k] - newMSE[k]) / newMSE[k] for k in range(imBands)], 6), dtype=np.float32)
        else:
            distortion = np.array([np.inf for k in range(imBands)], dtype=np.float32)
        MSE = newMSE

        print("Done!\nMSE:\t{}\nDistortion:\t{}".format(np.round(MSE, 6), distortion))
        print("###########################################")

        # Check enough convergency of MSE
        if abs(distortion[0]) < 0.005 and abs(distortion[1]) < 0.005 and abs(distortion[2]) < 0.005:
            print("Convergency on {} iterations -> exiting loop".format(iter + 1))
            break

    qCat = np.array(np.round(qCat, 0), dtype=np.uint8)
    rep = np.array(rep, dtype=np.uint8)

    print("Representation Values:\n{}".format(rep))
    print("###########################################")

    print("Creating file {}...".format(outCat), end='', flush=True)
    if not isfile(outCat):
        open(outCat, "x").close()
    f = open(outCat, "wb")
    print("done!")

    # 3x4 Bytes for dimensions + 1 Byte for nIntervals
    print("Inserting headers", end='', flush=True)
    for shape in cat.shape:  # Width, Height, Bands
        f.write(struct.pack("i", shape))
    print(".", end='', flush=True)
    f.write(np.array([nIntervals], dtype=np.uint8)[0])
    print(".", end='', flush=True)

    # 3x8 Bytes for representative values
    for k in range(imBands):
        for l in range(nIntervals):
            f.write(rep[k][l])
    print(".done!")

    count = 0
    # Map to representation values identifiers
    # On each band -> [0-8] -> 3 bits per sample
    print("Mapping to 3 bps...", end='', flush=True)
    fCat = np.zeros(cat.shape, dtype=np.uint8)
    for i in range(imWidth):
        for j in range(imHeight):
            for k in range(imBands):
                verif = 0
                for l in np.array(range(nIntervals), dtype=np.uint8):
                    if cat[i][j][k] > tre[k][l] \
                            and cat[i][j][k] <= tre[k][l + 1]:
                        qCat[i][j][k] = rep[k][l]
                        fCat[i][j][k] = l
                        verif = 1
                if verif == 0:
                    if cat[i][j][k] > - 1 and cat[i][j][k] < tre[k][0]:
                        qCat[i][j][k] = rep[k][0]
                        fCat[i][j][k] = 0
                    elif cat[i][j][k] < 256 and cat[i][j][k] > tre[k][nIntervals - 1]:
                        qCat[i][j][k] = rep[k][nIntervals - 1]
                        fCat[i][j][k] = nIntervals - 1
                    else:
                        qCat[i][j][k] = rep[k][0]
                        fCat[i][j][k] = 0
    print("done!")

    print("Adapting structures for bitmasking.", end='', flush=True)
    imComps = imWidth * imHeight * imBands
    newFCat = np.zeros([imComps], dtype=np.uint8)
    idx = 0
    for i in range(imWidth):
        for j in range(imHeight):
            for k in range(imBands):
                newFCat[idx] = fCat[i][j][k]
                idx += 1
    print(".", end='', flush=True)
    residual = (imComps % 24) // 3
    resres = imComps % 24
    bitPack = np.zeros([imComps], dtype=np.uint8) # 1D bitPacker
    print(".done!")
    """ The ideal mechanism is technically like this

          pixel A  |  | pixel B  |  | pixel C
        000 000 00|0  000 000 0|00  000 000
          byte A  |   byte B   |   byte C
        --------------------------------------
        C | | pixel  D |  |  pixel E |  | pixel F
        000 000 00|0 000  000 0|00 000  000
          byte D  |   byte E   |   byte F
        --------------------------------------
           F  |  | pixel  G |  | pixel  H |    pixel *I* -> A
        000 000  00|0 000 000  0|00 000 000
          byte G   |   byte H   |   byte I      byte *J* -> A

        If (width * height * bands) % 9 != 0:
            Special case: view 8 possible cases of filling
            8 -> 24 bits = 3 bytes  ->  000 000 00|0  000 000 0|00 000 000
            7 -> 21 bits = 3 bytes  ->  000 000 00|0  000 000 0|00 000 XXX
            6 -> 18 bits = 3 bytes  ->  000 000 00|0  000 000 0|00 XXX XXX
            5 -> 15 bits = 2 bytes  ->  000 000 00|0  000 000 X
            4 -> 12 bits = 2 bytes  ->  000 000 00|0  000 XXX X
            3 ->  9 bits = 2 bytes  ->  000 000 00|0  XXX XXX X
            2 ->  6 bits = 1 byte   ->  000 000 XX
            1 ->  3 bits = 1 byte   ->  000 XXX XX
    """
    print("Bit Packing", end='', flush=True)
    idx = 0
    for i in range(0, imComps - resres, 24):
        bitPack[idx + 0] = ((newFCat[i + 0] & 0b00000111) << 5) | ((newFCat[i + 1] & 0b00000111) << 2) | ((newFCat[i + 2] & 0b00000111) >> 1)  # A
        bitPack[idx + 1] = ((newFCat[i + 2] & 0b00000111) << 7) | ((newFCat[i + 3] & 0b00000111) << 4) | ((newFCat[i + 4] & 0b00000111) << 1) | ((newFCat[i + 5]  & 0b00000111) >> 2)  # B
        bitPack[idx + 2] = ((newFCat[i + 5] & 0b00000111) << 6) | ((newFCat[i + 6] & 0b00000111) << 3) | ((newFCat[i + 7] & 0b00000111))  # C

        bitPack[idx + 3] = ((newFCat[i + 8] & 0b00000111) << 5) | ((newFCat[i + 9] & 0b00000111) << 2) | ((newFCat[i +10] & 0b00000111) >> 1)  # D
        bitPack[idx + 4] = ((newFCat[i +10] & 0b00000111) << 7) | ((newFCat[i +11] & 0b00000111) << 4) | ((newFCat[i +12] & 0b00000111) << 1) | ((newFCat[i + 13] & 0b00000111) >> 2)  # E
        bitPack[idx + 5] = ((newFCat[i +13] & 0b00000111) << 6) | ((newFCat[i +14] & 0b00000111) << 3) | ((newFCat[i +15] & 0b00000111))  # F

        bitPack[idx + 6] = ((newFCat[i +16] & 0b00000111) << 5) | ((newFCat[i +17] & 0b00000111) << 2) | ((newFCat[i +18] & 0b00000111) >> 1)  # G
        bitPack[idx + 7] = ((newFCat[i +18] & 0b00000111) << 7) | ((newFCat[i +19] & 0b00000111) << 4) | ((newFCat[i +20] & 0b00000111) << 1) | ((newFCat[i + 21] & 0b00000111) >> 2)  # H
        bitPack[idx + 8] = ((newFCat[i +21] & 0b00000111) << 6) | ((newFCat[i +22] & 0b00000111) << 3) | ((newFCat[i +23] & 0b00000111))  # I
        idx += 9

    print(".", end='', flush=True)

    if residual > 5: # Write 3 bytes
        bitPack[idx + 0] = ((newFCat[imComps - resres - 1 + 0] & 0b00000111) << 5) | ((newFCat[imComps - resres - 1 + 1] & 0b00000111) << 2) | ((newFCat[imComps - resres + 2] & 0b00000111) >> 1) # A
        bitPack[idx + 1] = ((newFCat[imComps - resres - 1 + 2] & 0b00000111) << 7) | ((newFCat[imComps - resres - 1 + 3] & 0b00000111) << 4) | ((newFCat[imComps - resres + 4] & 0b00000111) << 1) | ((newFCat[imComps - resres + 5]  & 0b00000111) >> 2)  # B
        bitPack[idx + 2] = ((newFCat[imComps - resres - 1 + 5] & 0b00000111) << 6) | (((newFCat[imComps - resres - 1 + 6] & 0b00000111) << 3) if residual > 6 else 0b0) | (((newFCat[imComps - resres + 7] & 0b00000111)) if residual > 7 else 0b0)  # C
        idx += 3
    elif residual > 2: # Write 2 bytes
        bitPack[idx + 0] = ((newFCat[imComps - resres - 1 + 0] & 0b00000111) << 5) | ((newFCat[imComps - resres - 1 + 1] & 0b00000111) << 2) | ((newFCat[imComps - resres + 2] & 0b00000111) >> 1) # A
        bitPack[idx + 1] = ((newFCat[imComps - resres - 1 + 2] & 0b00000111) << 7) | (((newFCat[imComps - resres - 1 + 3] & 0b00000111) << 4) if residual > 3 else 0b0) | (((newFCat[imComps - resres + 4] & 0b00000111) << 1) if residual > 4 else 0b0) # B
        idx += 2
    elif residual > 0: # Write 1 byte
        bitPack[idx + 0] = ((newFCat[imComps - resres - 1 + 0] & 0b00000111) << 5) | (((newFCat[imComps - resres - 1 + 1] & 0b00000111) << 2) if residual == 2 else 0b0) # A
        idx += 1

    print(".", end='', flush=True)

    for i in range(idx):
        f.write(bitPack[i])
    print(".done!")

    print("Success on Quantization!")
    f.close()

    # Just for comparisons with decoder
    #imageio.imwrite("{}.png".format(outCat), qCat)

if __name__ == "__main__":
    main(sys.argv[1:])
