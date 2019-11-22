import sys
#from sifDataStructure import SIFTests
import numpy as np
# import time

class AlignMethod:
    def __init__(s):
        return

    # Split the long vector into matrices
    # of 502 * 501.
    def matrixSplit(s, matrix):
        return np.split(matrix, 502)

    # sideLength: the length of the side of the square block matrix (a x a)
    # leftCorner: a list [i, j] that defines the indices of the left corner of the block in the image matrix
    # need to throw error when sideLength + leftCorner > size of the matrix
    def findBlock(s, matrix, sideLength, leftCorner):
        i = 0
        block = []
        while i < sideLength:
            each = matrix[i + leftCorner[0]]
            each = each[leftCorner[1]:(leftCorner[1] + sideLength)].tolist()
            block.append(each)
            i = i + 1
        return np.array(block)

    # Compute element-wise multiplications and a final summation of two given matrices
    # blkRef and blkAlg: two given matrices
    def matrixDotProduct(s, blkRef, blkAlg):
        return np.sum(np.multiply(blkRef, blkAlg))

    # Compute the mean image of a given image; i.e. compute the mean intensity
    # of a given image (or a portion of image)
    # block: the image matrix
    # sideLenth: the length of the side of the block.
    def meanImage(s, block, sideLength):
        numPix = sideLength * sideLength
        return (np.sum(block) / numPix)

    # Compute the normalized cross correlation score of two given matrices
    # blkRef and blkAlg: two given matrices
    # sideLength: length of the side of the given matrices
    def normalizedCrossCorrelation(s, blkRef, blkAlg, sideLength):
        meanRef = s.meanImage(blkRef, sideLength)
        meanAlg = s.meanImage(blkAlg, sideLength)
        blkRefNorm = blkRef - meanRef
        blkAlgNorm = blkAlg - meanAlg
        blkRefSq = np.square(blkRefNorm)
        blkAlgSq = np.square(blkAlgNorm)
        Encc = s.matrixDotProduct(blkRefNorm, blkAlgNorm) / np.sqrt(s.matrixDotProduct(blkRefSq, blkAlgSq))
        return Encc

    def findMaxCoor(s, blkRef, matrix, sideLength, refCoor):
        maxCoor = []
        searchBox = 10
        max = 0
        for row in list(range(refCoor[0] - searchBox, refCoor[0]+ (1 * searchBox))):
            for column in list(range(refCoor[1] - searchBox, refCoor[1]+ (1 * searchBox))):
                blkCompare = s.findBlock(matrix, sideLength, [row, column])
                normalizedScore = s.normalizedCrossCorrelation(blkRef, blkCompare, sideLength)
                if normalizedScore > max:
                    max = normalizedScore
                    maxCoor = [row, column]
        # print("Maximum Coordinate is located. ")
        return maxCoor

    # Translate the image based on the given displacement vector.
    # blkMove: the translated image; displacement: a vector
    def translation(s, blkMove, displacement):
        # vertical translation
        if displacement[0] != 0:
            blkMove = np.roll(blkMove, -displacement[0], axis = 0)
        if displacement[1] != 0:
            blkMove = np.roll(blkMove, -displacement[1], axis = 1)
        return blkMove

    # Align the atom image based on the reference image by computing the NCC score
    # of the chosen patch of size (defined by sideLength),
    # which is found using the reference coordinate refCoor.
    # atoms: the atom image; ref: the reference image; sideLength: length of the sides of the square
    def alignment(s, atoms, ref, sideLength, refCoor):
        #start2 = time.time()
        matrixAlg = np.array(s.matrixSplit(atoms))
        matrixRef = np.array(s.matrixSplit(ref))
        center = s.findBlock(matrixRef, sideLength, refCoor)
        maxCoor = s.findMaxCoor(center, matrixAlg, sideLength, refCoor)
        s.trans = [maxCoor[0] - refCoor[0], maxCoor[1] - refCoor[1]]
        print("Displacement is " + str(s.trans))
        alignedMatrix = s.translation(matrixAlg, s.trans)
        #end2 = time.time()
        #print("Time: " + str((end2 - start2)) + " seconds.")
        return alignedMatrix

    # compute normalization
    def processAlg(s, aligned, ref, background):
        '''
        start = time.time()
        alignedLong = []
        for arr in aligned:
            for element in arr:
                alignedLong.append(element)
        alignedLong = np.array(alignedLong)
        end = time.time()
        print("Time: " + str((end - start)) + " seconds.")
        '''
        alignedLong = aligned.flatten()
        """Given a background and reference shot, calculates the absorption image"""
        #start2 = time.time()
        s.final = (alignedLong-background+0.000001)/(ref-background+0.000001)
        #end2 = time.time()
        #print("Time: " + str((end2 - start2)) + " seconds.")

if __name__ == "__main__":
    print("This module is not directly runnable")
