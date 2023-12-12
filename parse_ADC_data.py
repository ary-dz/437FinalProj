import numpy as np
import os
import struct
import scipy.io as sio


# set file paths
# CHANGE THESE TO MATCH YOUR SETUP

binFileName = "pHistBytes"
number_of_files = 6
numAntennas = 3

# declaring TLV types compatible for ADC data stream
MMWDEMO_OUTPUT_EXT_MSG_ADC_SAMPLES = 316

# Set UART TLV Structures
hex_values = np.array([0x0102, 0x0304, 0x0506, 0x0708], dtype=np.uint16)
syncPatternUINT64 = hex_values.view(np.uint64)
syncPatternUINT8 = hex_values.view(np.uint8)

# defining header structure
frameHeaderStructType = np.dtype([
    ('sync', np.uint64),
    ('version', np.uint32),
    ('packetLength', np.uint32),
    ('platform', np.uint32),
    ('frameNumber', np.uint32),
    ('timeinCPUcycles', np.uint32),
    ('numDetectObject', np.uint32),
    ('numTLVs', np.uint32),
    ('subframe', np.uint32)
])

tlvHeaderStruct = np.dtype([
    ('type', np.uint32),
    ('length', np.uint32)
])

# header length
frameHeaderLengthInBytes = frameHeaderStructType.itemsize
tlvHeaderLengthInBytes = tlvHeaderStruct.itemsize

# Frame history structure
frameStatStruct = np.dtype([
    ('targetFrameNum', np.uint32),
    ('header', frameHeaderStructType),
    ('bytes', np.uint32),
    ('adcSamples', np.int16, (numAntennas,256))
])

# Function to read the frame header
def readFrameHeader(fid, frameHeaderLengthInBytes, syncPatternUINT8):
    lostSync = True
    outOfSyncBytes = 0
    while lostSync:
        syncPatternFound = True
        for n in range(8):
            rxByte = fid.read(1)
            if not rxByte:
                return None, 0, 0  # End of file
            rxByte = struct.unpack('<B', rxByte)[0]
            if rxByte != syncPatternUINT8[n]:
                syncPatternFound = False
                outOfSyncBytes += 1
                break
        if syncPatternFound:
            lostSync = False
            header = fid.read(frameHeaderLengthInBytes - 8)
            header = struct.unpack(f'<{frameHeaderLengthInBytes - 8}B', header)
            header = np.asarray(header, dtype=np.uint8)
            header = np.concatenate((syncPatternUINT8, header))
            # header = struct.unpack('<Q', syncPatternUINT8 + header)[0]
            return header, frameHeaderLengthInBytes, outOfSyncBytes
        
def parse_ADC(binFilePath, output_file=None):
    toreturn = np.array([], dtype=frameStatStruct) #file to be returned
    targetFrameNum = 0

    # Open and read using the method from the visualizer
    for file_num in range(1, number_of_files + 1):
        thisBinFileName = os.path.join(binFilePath, f"{binFileName}_{file_num}.bin")
        print(f"processing file: {thisBinFileName}")

        #open file and read al the ADC data
        with open(thisBinFileName, 'rb') as fid:
            lostSync = False
            gotHeader = False
            
            while True:
                fHist = np.zeros(1, dtype=frameStatStruct)
                if not gotHeader:
                    # Read the header first
                    rxHeader, byteCount, outOfSyncBytes = readFrameHeader(fid, frameHeaderLengthInBytes, syncPatternUINT8)
                    gotHeader = True

                # Double check the header size
                if byteCount != frameHeaderLengthInBytes:
                    reason = 'Header Size is wrong'
                    lostSync = True
                    break

                # Double check the sync pattern
                magicBytes = np.frombuffer(rxHeader[:8], dtype=np.uint64)
                if not np.array_equal(magicBytes, syncPatternUINT64):
                    reason = 'No SYNC pattern'
                    lostSync = True
                    break

                # define temporaly data structure
                fHist_temp = np.zeros(1, dtype=frameStatStruct)

                # Parse the header
                frameHeader = np.frombuffer(rxHeader, dtype=frameHeaderStructType)

                if gotHeader:
                    print(f"frame number is: {frameHeader['frameNumber'][0]}")
                    if frameHeader['frameNumber'][0] >= targetFrameNum:
                        # We have a valid header
                        targetFrameNum = frameHeader['frameNumber'][0]
                        frameNum = frameHeader['frameNumber'][0]
                        if outOfSyncBytes > 0:
                            print(f'Found sync at frame {targetFrameNum}({frameNum}). Discarded out of sync bytes: {outOfSyncBytes}')
                        gotHeader = False
                    else:
                        reason = 'Old Frame'
                        gotHeader = False
                        lostSync = True
                        break
                
                # Start processing the header
                fHist['targetFrameNum'] = targetFrameNum
                fHist['header'] = frameHeader
                dataLength = frameHeader['packetLength'] - frameHeaderLengthInBytes
                fHist['bytes'] = dataLength

                adcSamples = []
                if dataLength[0] > 0:
                    # Read all packet
                    rxData = np.fromfile(fid, dtype=np.uint8, count=dataLength[0])
                    if rxData.size != dataLength:
                        reason = 'Data Size is wrong'
                        lostSync = True
                        break
                    offset = 0

                    # TLV Parsing
                    for nTlv in range(frameHeader['numTLVs'][0]):
                        tlvType = np.frombuffer(rxData[offset:offset+4], dtype=np.uint32)[0]
                        tlvLength = np.frombuffer(rxData[offset+4:offset+8], dtype=np.uint32)[0]
                        if tlvLength + offset > dataLength:
                            reason = 'TLV Size is wrong'
                            lostSync = True
                            break
                        offset += tlvHeaderLengthInBytes
                        valueLength = tlvLength
                    
                        if tlvType == MMWDEMO_OUTPUT_EXT_MSG_ADC_SAMPLES:
                            adcSamples = np.frombuffer(rxData[offset:offset+valueLength], dtype=np.int16)
                            adcSamples = np.reshape(adcSamples, (-1, numAntennas)).T
                            offset += valueLength
                        else:
                            reason = 'TLV Type is wrong or not supported'
                            lostSync = True
                            break

                    if lostSync:
                        print(reason)
                        break
                # Store ADC data
                fHist['adcSamples'] = adcSamples
                toreturn = np.append(toreturn, fHist)
    if output_file:
        np.save(output_file, toreturn, allow_pickle=True)
    return toreturn
    

if __name__ == "__main__":
    binFilePath = "/Users/ryuokubo/Library/CloudStorage/Box-Box/CS437_TI_radar/Industrial_Visualizer/binData/09_07_2023_09_00_19/"

    toreturn = np.array([], dtype=frameStatStruct) #file to be returned
    targetFrameNum = 0

    # Open and read using the method from the visualizer
    for file_num in range(1, number_of_files + 1):
        thisBinFileName = os.path.join(binFilePath, f"{binFileName}_{file_num}.bin")
        print(f"processing file: {thisBinFileName}")

        #open file and read al the ADC data
        with open(thisBinFileName, 'rb') as fid:
            lostSync = False
            gotHeader = False
            
            while True:
                fHist = np.zeros(1, dtype=frameStatStruct)
                if not gotHeader:
                    # Read the header first
                    rxHeader, byteCount, outOfSyncBytes = readFrameHeader(fid, frameHeaderLengthInBytes, syncPatternUINT8)
                    gotHeader = True

                # Double check the header size
                if byteCount != frameHeaderLengthInBytes:
                    reason = 'Header Size is wrong'
                    lostSync = True
                    break

                # Double check the sync pattern
                magicBytes = np.frombuffer(rxHeader[:8], dtype=np.uint64)
                if not np.array_equal(magicBytes, syncPatternUINT64):
                    reason = 'No SYNC pattern'
                    lostSync = True
                    break

                # define temporaly data structure
                fHist_temp = np.zeros(1, dtype=frameStatStruct)

                # Parse the header
                frameHeader = np.frombuffer(rxHeader, dtype=frameHeaderStructType)

                if gotHeader:
                    print(f"frame number is: {frameHeader['frameNumber'][0]}")
                    if frameHeader['frameNumber'][0] >= targetFrameNum:
                        # We have a valid header
                        targetFrameNum = frameHeader['frameNumber'][0]
                        frameNum = frameHeader['frameNumber'][0]
                        if outOfSyncBytes > 0:
                            print(f'Found sync at frame {targetFrameNum}({frameNum}). Discarded out of sync bytes: {outOfSyncBytes}')
                        gotHeader = False
                    else:
                        reason = 'Old Frame'
                        gotHeader = False
                        lostSync = True
                        break
                
                # Start processing the header
                fHist['targetFrameNum'] = targetFrameNum
                fHist['header'] = frameHeader
                dataLength = frameHeader['packetLength'] - frameHeaderLengthInBytes
                fHist['bytes'] = dataLength

                adcSamples = []
                if dataLength[0] > 0:
                    # Read all packet
                    rxData = np.fromfile(fid, dtype=np.uint8, count=dataLength[0])
                    if rxData.size != dataLength:
                        reason = 'Data Size is wrong'
                        lostSync = True
                        break
                    offset = 0

                    # TLV Parsing
                    for nTlv in range(frameHeader['numTLVs'][0]):
                        tlvType = np.frombuffer(rxData[offset:offset+4], dtype=np.uint32)[0]
                        tlvLength = np.frombuffer(rxData[offset+4:offset+8], dtype=np.uint32)[0]
                        if tlvLength + offset > dataLength:
                            reason = 'TLV Size is wrong'
                            lostSync = True
                            break
                        offset += tlvHeaderLengthInBytes
                        valueLength = tlvLength
                    
                        if tlvType == MMWDEMO_OUTPUT_EXT_MSG_ADC_SAMPLES:
                            adcSamples = np.frombuffer(rxData[offset:offset+valueLength], dtype=np.int16)
                            adcSamples = np.reshape(adcSamples, (-1, numAntennas)).T
                            offset += valueLength
                        else:
                            reason = 'TLV Type is wrong or not supported'
                            lostSync = True
                            break

                    if lostSync:
                        print(reason)
                        break
                # Store ADC data
                fHist['adcSamples'] = adcSamples
                toreturn = np.append(toreturn, fHist)

    np.save("test.npy", toreturn, allow_pickle=True)
