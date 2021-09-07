import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import warnings


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CamelyonDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, slides_root_dir, transform, is_training, max_bag_size, seed=123):
        """
        :param slides_folders: list of abs paths of slide folder (which should contains images, summary/label/percent
            files
        :param is_training: True if is training, else False (for data augmentation)
        :param max_bag_size: maximum number of instances to be returned per bag
        """
        self.is_training = is_training
        slides = []
        for root, _, files in os.walk(slides_root_dir, topdown=False):
            if any('metadata.txt' in file_name for file_name in files):
                if self.is_training and 'training' in root:
                    slides.append(root)
                elif not self.is_training and 'test' in root:
                    slides.append(root)
        self.slides_folders = np.array(slides)
        self.max_bag_size = max_bag_size

        self.slides_ids = []  # ids slides
        self.slides_labels = []  # raw str labels
        self.slides_images_filepaths = []  # list of all in-dataset tilespaths of slides
        self.all_images = []  # all tiles to be used in the dataset, gets resampled in each epoch
    
        self.transform = transform
        self.seed = seed

        slides_ids, slides_labels, slides_images_filepaths, all_images = self.load_data()
        self.slides_ids = np.array(slides_ids)
        self.slides_labels = np.array(slides_labels)
        self.slides_images_filepaths = np.array(slides_images_filepaths)
        self.all_images = np.array(all_images)

        assert len(self.slides_ids) == len(self.slides_labels) == len(self.slides_images_filepaths), \
            'mismatch in slides containers lengths %s' % (
            ' '.join(str(len(l)) for l in [self.slides_ids, self.slides_labels,
                                           self.slides_images_filepaths]))
        assert len(all_images) == len(self.slides_ids)*self.max_bag_size
        if self.is_training:
            self._resample_all_slides()  # Shuffle slides

    def load_data(self):
        slides_ids, slides_labels, slides_images_filepaths, all_images = [], [], [], []

        # Name of expected non-image files for all slides folders
        metadata_filename = 'metadata.txt'
        reference_file = None  # Only used for test dataset

        # Seek all slides folders, and load static data including list of tiles filepaths and bag label
        for slide_folder in self.slides_folders:
            all_slide_files = list(filter(lambda f: os.path.isfile(os.path.join(slide_folder, f)),
                                          [x for x in os.listdir(slide_folder)]))

            # Seek and save label, case_id and summary files: expects 1 and only 1 for each
            for data_filename in [metadata_filename]:
                assert sum([f == data_filename for f in all_slide_files]) == 1, \
                    'slide %s: found %d files for %s, expected 1' % (slide_folder,
                                                                     sum([f == data_filename for f in
                                                                          all_slide_files], ),
                                                                     data_filename)

            metadata_file = os.path.join(slide_folder, [f for f in all_slide_files if f == metadata_filename][0])
            with open(metadata_file, 'r') as f:
                metadata_file = f.read()

            # Seek all filtered images of slide (not-background images)
            slide_images_filenames = list(filter(lambda f: f.endswith(('.jpeg', '.jpg', '.png')), all_slide_files))

            if len(slide_images_filenames) == 0:
                warnings.warn('No image in ' + os.path.basename(slide_folder))
                continue
            
            if self.is_training:
                label = 0 if 'normal' in os.path.basename(slide_folder) else 1
            else:
                if reference_file is None:
                    reference_file = open(os.path.join(os.path.dirname(os.path.abspath(slide_folder)), 'reference.csv')).read().split('\n')
                slide_label = [current_line.split(',')[1] for current_line in reference_file if slide_folder.split('/')[-1] in current_line][0]
                label = 0 if slide_label == 'Normal' else 1
            # Save data
            slides_ids.append(os.path.basename(slide_folder))
            slides_labels.append(label)
            images_in_folder = list(map(lambda f: os.path.abspath(os.path.join(slide_folder, f)), slide_images_filenames))
            if self.max_bag_size > len(images_in_folder):
                all_images.extend(random.choices(images_in_folder, k=self.max_bag_size))
            else:
                all_images.extend(random.sample(images_in_folder, self.max_bag_size))
            slides_images_filepaths.append(images_in_folder)

        slides_ids = slides_ids
        slides_labels = np.asarray(slides_labels)

        return slides_ids, slides_labels, slides_images_filepaths, all_images

    def show_bag(self, bag_idx, savefolder=None):
        """ Plot/save tiles sampled from the slide of provided index """
        bag = self._get_slide_instances(bag_idx)
        bag_label = self.slides_labels[bag_idx]
        tr = transforms.ToTensor()
        bag = [tr(b) for b in bag]
        imgs = make_grid(bag)

        npimgs = imgs.numpy()
        plt.imshow(np.transpose(npimgs, (1, 2, 0)), interpolation='nearest')
        plt.title('Bag label: %s | %d instances' % (bag_label, len(bag)))
        if savefolder is not None:
            plt.savefig(os.path.join(savefolder, 'show_' + str(bag_idx) + '.png'), dpi=1000)
        else:
            plt.show()

    def _get_slide_instances(self, item):
        """ Memory load all tiles or randomly sampled tiles from slide of specified index """
        slide_images_filepaths = self.slides_images_filepaths[item]

        # Randomly sample the specified max number of tiles from the slide with replacement
        if self.max_bag_size is not None:
            slide_images_filepaths = random.choices(slide_images_filepaths, k=self.max_bag_size)

        # Load images
        bag_images = [pil_loader(slide_image_filepath) for slide_image_filepath in slide_images_filepaths]
        return bag_images

    def _resample_all_slides(self):
        print("Resampling every slide")
        indices = np.arange(len(self.slides_folders))
        np.random.shuffle(indices)
        self.slides_ids = self.slides_ids[indices]
        self.slides_labels = self.slides_labels[indices]
        self.slides_images_filepaths = self.slides_images_filepaths[indices]
        self.all_images = []
        for bag_idx in range(len(self.slides_ids)):
            if self.max_bag_size > len(self.slides_images_filepaths[bag_idx]):
                self.all_images.extend(random.choices(self.slides_images_filepaths[bag_idx], k=self.max_bag_size))
            else:
                self.all_images.extend(random.sample(self.slides_images_filepaths[bag_idx], self.max_bag_size))

    def __getitem__(self, item):
        if item == 0 and self.is_training:
            self._resample_all_slides()
        return self.transform(pil_loader(self.all_images[item])), self.slides_labels[int(item/self.max_bag_size)], self.slides_ids[int(item/self.max_bag_size)]

    def __len__(self):
        return len(self.all_images)