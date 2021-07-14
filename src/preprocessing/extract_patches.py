import argparse
import large_image
import os
import numpy as np
from openslide import open_slide
from skimage.filters import threshold_otsu
from PIL import Image
from tqdm import tqdm


def get_parser():
    """Returns args parser"""
    parser = argparse.ArgumentParser(description='Data processing module: Divides Camelyon16 dataset into patches')
    parser.add_argument('source_slides_folder', type=str, default=None,
                        help='Source folder containing slides')
    parser.add_argument('output_folder', type=str, default=None,
                        help='Folder to save resulting background filtered patches')
    parser.add_argument('--magnification', type=int, default=20,
                        help='Amount of magnification applied to the WSI for dividing into patches')
    parser.add_argument('--size', type=int, default=224,
                        help='Dimension of the patches in pixels')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Amount of overlap between neighboring patches in ratio')
    parser.add_argument('--tissue-threshold', type=float, default=0.75,
                        help='Amount of tissue in a patch to threshold')
    parser.add_argument('--threshold-rgb', action='store_true', default=False,
                        help='Apply otsu threshold on RGB channels instead of grayscale')
    return parser


def discover_slides(folder):
    """Walks through all files in the given folder and subfolders, returns a list of slide paths.

    Parameters
    ----------
    folder : string
        Main folder of dataset

    Returns
    -------
    slides : list
        List of slide paths found

    """
    slides = []
    for root, _, files in os.walk(folder, topdown=False):
        for file_name in files:
            if ".tif" in file_name:
                slides.append(os.path.join(root, file_name))
    return slides


def get_threshold(slide, rgb):
    """Calculates the otsu threshold level of a WSI.

    Parameters
    ----------
    slide : OpenSlide
        Whole Slide Image.
    rgb : bool
        Whether the otsu threshold be calculated in RGB channels or in grayscale image.

    Returns
    -------
    thresholds : list
        List of thresholds in respective channels.

    """
    thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))  #: Thumbnail slide
    thresholds = []
    if rgb:
        thumbnail_arr = np.asarray(thumbnail)
        thresholds.extend([threshold_otsu(thumbnail_arr[:, :, 0]),   #: R
                           threshold_otsu(thumbnail_arr[:, :, 1]),   #: G
                           threshold_otsu(thumbnail_arr[:, :, 2])])  #: B
    else:
        thumbnail_arr = np.asarray(thumbnail.convert('L'))  #: Grayscale
        thresholds.append(threshold_otsu(thumbnail_arr))
    return thresholds


def tile_slide(slide_path, out_folder, size, overlap, magnification, tissue_threshold, thresholds):
    """Divides slide into patches, filters background and saves tissue images.

    Parameters
    ----------
    slide_path : string
        Whole Slide Image path.
    out_folder : string
        Path for saving patch images.
    size : int
        Size of the patches in pixels. Patches will be square.
    overlap : int
        Amount of overlap in pixels.
    magnification : int
        Magnification applied to Whole Slide Image.
    tissue_threshold : float
        Amount of tissue in a patch in order to save it.
    thresholds : list
        List of otsu thresholds in respective channels.

    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file = open(os.path.join(out_folder, 'metadata.txt'), 'a+')  #: Generate if file doesn't exist
    file.write('Patch Name, Global X position, Global X width, Global Y position, Global Y Height, Tissue Ratio\n')
    ts = large_image.getTileSource(slide_path)
    iteratorArgs = {
        "scale": dict(magnification=magnification),
        "tile_size": dict(width=size, height=size),
        "tile_overlap": dict(x=overlap, y=overlap),
        "format": large_image.tilesource.TILE_FORMAT_NUMPY
    }
    tile_count = ts.getTileCount(**iteratorArgs)  #: To print progression
    ts_iterator = ts.tileIterator(**iteratorArgs)  #: Generate an iterator for tiling
    for tile in tqdm(ts_iterator, total=tile_count):
        img_array = tile['tile']  #: Get image as numpy array
        mask = np.ones((img_array.shape[0], img_array.shape[1]))
        if len(thresholds) > 1:  #: Apply threshold on RGB channels
            for i, img_channel in enumerate([img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]]):
                channel_mask = img_channel < thresholds[i]
                channel_mask = np.array(channel_mask, dtype=int)
                mask = mask * channel_mask
        else:  #: Apply threshold on grayscale channel
            gray_img = np.asarray(Image.fromarray(img_array).convert('L'))  #: Convert to grayscale
            channel_mask = gray_img < thresholds[0]
            mask = np.array(channel_mask, dtype=int)
        tissue_ratio = np.sum(mask)/mask.size
        if tissue_ratio > tissue_threshold:
            im = Image.fromarray(img_array)
            img_name = str(tile['tile_x']) + "_" + str(tile['tile_y']) + ".png"
            im.save(os.path.join(out_folder, img_name))
            file.write(img_name + ',' + str(int(tile['gx'])) + ',' + str(int(tile['gwidth'])) + ',' +
                       str(int(tile['gy'])) + ',' + str(int(tile['gheight'])) + ',' + str(tissue_ratio) + '\n')
            file.flush()
    file.close()


def get_processed_slides(folder_dir):
    """Opens txt file and parses processed slides information

    Parameters
    ----------
    folder_dir : string
        txt folder path

    Returns
    -------
    file : TextIOWrapper
        txt file
    processed_slides : list
        List of processed slides and magnification

    """
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    file = open(os.path.join(folder_dir, 'processed_slides.txt'), 'a+')  #: Generate if file doesn't exist
    file.seek(0)  #: Move pointer to the first line
    processed_slides = file.read()
    processed_slides = processed_slides.split('\n')
    return file, processed_slides[:-1]


def _main(args):
    file, processed_slides = get_processed_slides(args.output_folder)
    slides = discover_slides(args.source_slides_folder)
    print(f'Discovered {len(slides)} slides in total')
    for slide_path in tqdm(slides[:1]):
        if slide_path + ',' + str(args.magnification) in processed_slides:  #: If we already processed the slide before
            continue
        slide = open_slide(slide_path)
        thresholds = get_threshold(slide, args.threshold_rgb)
        overlap = int(args.size * args.overlap)
        slide_name = slide_path.split('/')[-1]
        out_folder = ""
        if "normal" in slide_name:
            out_folder = os.path.join(args.output_folder, 'mag' + str(args.magnification), 'training', slide_name[:-4])
        elif "tumor" in slide_name:
            out_folder = os.path.join(args.output_folder, 'mag' + str(args.magnification), 'training', slide_name[:-4])
        else:
            out_folder = os.path.join(args.output_folder, 'mag' + str(args.magnification), 'test', slide_name[:-4])
        print(f'Processing slide {slide_name}')
        tile_slide(slide_path, out_folder, args.size, overlap, args.magnification, args.tissue_threshold, thresholds)
        file.write(slide_path + ',' + str(args.magnification) + '\n')
        file.flush()  #: Write to txt before closing the file
    file.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    #: Print out information
    print(f'WSI Dataset is in {args.source_slides_folder}')
    print(f'Processed patches will be put into {args.output_folder}')
    print(f'WSI will be looked at {args.magnification}x magnification')
    print(f'Patches will be of size {args.size}x{args.size} with {args.overlap*100}% overlap')
    print(f'Patch will be kept if it contains {args.tissue_threshold * 100}% tissue')
    print(f'Otsu threshold will be applied in {3 if args.threshold_rgb else 1} channels')
    _main(args)
