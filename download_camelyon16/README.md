# Download Camelyon16 Dataset

You can either download images of Camelyon16 using Google Drive (faster option) or Giga Science.

In order to download from Google Drive, run

`python download_grive.py /path/to/Camelyon16`

Note that Google Drive option will take note of so far downloaded slides, this way you can continue where you left off.

If you get Quota exceeded error from Google Drive, you may download from Giga Science by

`sh giga_science.sh /path/to/Camelyon16`

Both options will generate folders for you.

Lastly, download checksum.md5 file manually, navigate to the folder where checksum.md5 belongs to and run

`python checksum_test.py /path/to/Camelyon16`

This code checks the md5 hashes of downloaded images with the ground truth. If there is any corrupted image, you can pinpoint it and downloaded it manually.
