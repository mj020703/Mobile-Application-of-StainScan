import rawpy
import cv2
import numpy as np

with rawpy.imread("DataSet/StainScan/dataset/Ballpen Ink_cotton/IMG_6065.CR2") as raw:
    thumb = raw.extract_thumb()
    if thumb.format == rawpy.ThumbFormat.JPEG:
        arr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("scratch/sample_grey_cr2.jpg", arr)
        print("Saved thumbnail")
