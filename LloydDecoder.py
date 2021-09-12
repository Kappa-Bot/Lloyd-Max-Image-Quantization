import numpy as np
import imageio

import sys
from os.path import isfile
import struct

"""
    Beforehand:
    Sorry for the huge bitmasking tryhard code below
"""

def main(argv):
    if len(argv) != 2:
        print("Usage: cat_decoder.py <input_file> <output_file>")
        sys.exit(1)
    inCat, outCat = argv[0], argv[1]
    if not isfile(inCat):
        print("{} is not a valid file".format(inCat))
        sys.exit(2)
    print("Going to decode {} to output file {}".format(inCat, outCat))

    print("Reading file {}...".format(inCat), end='', flush=True)
    f = open(inCat, "rb")
    print("done!")

    print("Detecting...", end='', flush=True)
    dCat = np.zeros(np.array([struct.unpack('i', f.read(4))[0], struct.unpack('i', f.read(4))[0], struct.unpack('i', f.read(4))[0]], dtype=np.int32), dtype=np.uint8)
    imWidth, imHeight, imBands = dCat.shape[0], dCat.shape[1], dCat.shape[2]
    imComp = imWidth * imHeight * imBands
    print("Shape {}x{}x{}...".format(dCat.shape[0], dCat.shape[1], dCat.shape[2]), end='', flush=True)

    nIntervals = struct.unpack('B', f.read(1))[0]
    R = np.zeros([imBands, nIntervals], dtype=np.uint8)
    print("{} Representation levels!".format(R.shape[1]))

    print("Extracting Representation values...", end='', flush=True)
    for k in range(imBands):
        for l in range(nIntervals):
            R[k][l] = struct.unpack('B', f.read(1))[0]
    print("done!")
    print(R)

    print("Unpacking bit bundle", end='', flush=True)
    residual = (imComp % 24) // 3
    resres = imComp % 24
    bitUnpacker = np.zeros([imComp], dtype=np.uint8)
    dCat1D = np.zeros([imComp], dtype=np.uint8)
    print(".", end='', flush=True)
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

    maxLen = 0
    while True:
        byte = f.read(1)
        if not byte:
            mark = maxLen
            f.close()
            break
        bitUnpacker[maxLen] = struct.unpack('B', byte)[0]
        maxLen += 1
    print(".", end='', flush=True)

    idx = 0
    for i in range(0, imComp - resres, 24):
        dCat1D[i + 0] = (bitUnpacker[idx + 0] & 0b11100000) >> 5
        dCat1D[i + 1] = (bitUnpacker[idx + 0] & 0b00011100) >> 2

        dCat1D[i + 2] = ((bitUnpacker[idx + 0] & 0b00000011) << 1) | ((bitUnpacker[idx + 1] & 0b10000000) >> 7)

        dCat1D[i + 3] = (bitUnpacker[idx + 1] & 0b01110000) >> 4
        dCat1D[i + 4] = (bitUnpacker[idx + 1] & 0b00001110) >> 1

        dCat1D[i + 5] = ((bitUnpacker[idx + 1] & 0b00000001) << 2) | ((bitUnpacker[idx + 2] & 0b11000000) >> 6)

        dCat1D[i + 6] = (bitUnpacker[idx + 2] & 0b00111000) >> 3
        dCat1D[i + 7] = (bitUnpacker[idx + 2] & 0b00000111)

        dCat1D[i + 8] = (bitUnpacker[idx + 3] & 0b11100000) >> 5
        dCat1D[i + 9] = (bitUnpacker[idx + 3] & 0b00011100) >> 2

        dCat1D[i +10] = ((bitUnpacker[idx + 3] & 0b00000011) << 1) | ((bitUnpacker[idx + 4] & 0b10000000) >> 7)

        dCat1D[i +11] = (bitUnpacker[idx + 4] & 0b01110000) >> 4
        dCat1D[i +12] = (bitUnpacker[idx + 4] & 0b00001110) >> 1

        dCat1D[i +13] = ((bitUnpacker[idx + 4] & 0b00000001) << 2) | ((bitUnpacker[idx + 5] & 0b11000000) >> 6)

        dCat1D[i +14] = (bitUnpacker[idx + 5] & 0b00111000) >> 3
        dCat1D[i +15] = (bitUnpacker[idx + 5] & 0b00000111)

        dCat1D[i +16] = (bitUnpacker[idx + 6] & 0b11100000) >> 5
        dCat1D[i +17] = (bitUnpacker[idx + 6] & 0b00011100) >> 2

        dCat1D[i +18] = ((bitUnpacker[idx + 6] & 0b00000011) << 1) | ((bitUnpacker[idx + 7] & 0b10000000) >> 7)

        dCat1D[i +19] = (bitUnpacker[idx + 7] & 0b01110000) >> 4
        dCat1D[i +20] = (bitUnpacker[idx + 7] & 0b00001110) >> 1

        dCat1D[i +21] = ((bitUnpacker[idx + 7] & 0b00000001) << 2) | ((bitUnpacker[idx + 8] & 0b11000000) >> 6)

        dCat1D[i +22] = (bitUnpacker[idx + 8] & 0b00111000) >> 3
        dCat1D[i +23] = (bitUnpacker[idx + 8] & 0b00000111)
        idx += 9


    print(".{}.{}.{}.{}".format(maxLen, idx, resres, residual), end='', flush=True)

    add = 0
    if residual > 0:
        dCat1D[imComp - resres + 0] = (bitUnpacker[idx + 0] & 0b11100000) >> 5
        add += 1
        if residual > 1:
            dCat1D[imComp - resres + 1] = (bitUnpacker[idx + 0] & 0b00011100) >> 2
            if residual > 2:
                add += 1
                dCat1D[imComp - resres + 2] = ((bitUnpacker[idx + 0] & 0b00000011) << 1) | ((bitUnpacker[idx - 1 + 1] & 0b10000000) >> 7)
                if residual > 3:
                    dCat1D[imComp - resres + 3] = (bitUnpacker[idx + 1] & 0b11100000) >> 5
                    if residual > 4:
                        dCat1D[imComp - resres + 4] = (bitUnpacker[idx + 1] & 0b00011100) >> 2
                        if residual > 5:
                            add += 1
                            dCat1D[imComp - resres + 5] =  ((bitUnpacker[idx + 1] & 0b00000011) << 1) | ((bitUnpacker[idx - 1 + 2] & 0b10000000) >> 7)
                            if residual > 6:
                                dCat1D[imComp - resres + 6] = (bitUnpacker[idx + 2] & 0b11100000) >> 5
                                if residual > 7:
                                    dCat1D[imComp - resres + 7] = (bitUnpacker[idx + 2] & 0b00011100) >> 2
    idx += add

    print("done!")
    print("Writing result on file {}...".format(outCat), end='', flush=True)
    if not isfile(outCat):
        open(outCat, "x").close()
    print("done!")

    # Direct map to the representative value
    # Replacement via index (3 bits ID)
    idx = 0
    for i in range(imWidth):
        for j in range(imHeight):
            for k in range(imBands):
                for l in range(nIntervals):
                    if R[k][l] == R[k][dCat1D[idx]]:
                        dCat[i][j][k] = R[k][dCat1D[idx]]
                        break
                idx += 1
    imageio.imwrite(outCat, dCat)
    print("Success on dequantization!")

if __name__ == "__main__":
    main(sys.argv[1:])