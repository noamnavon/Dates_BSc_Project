import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment_date(image, show_image=False):
    scales = [0.01, 0.05, 0.2]
    mask = None

    for scale in scales:
        small = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w = small.shape[:2]
        rect = (
            int(w // 2 - 1000 * scale),
            int(h // 2 - 1300 * scale),
            int(2000 * scale),
            int(3000 * scale)
        )
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        if mask is None:
            mask = np.zeros(small.shape[:2], np.uint8)
            cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
        else:
            mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cv2.grabCut(small, mask, (0, 0, 1, 1), bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
        if show_image:
            final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            plt.imshow(cv2.cvtColor(small * final_mask[:, :, np.newaxis], cv2.COLOR_BGR2RGB))
            plt.show()
            plt.imshow(mask)
            plt.show()

    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    final_mask = cv2.resize(final_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    segmented = image * final_mask[:, :, np.newaxis]
    return segmented
