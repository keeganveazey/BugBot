from undouble import Undouble
import os
import glob
import time
import os
from PIL import Image, ImageEnhance
import numpy as np
from collections import defaultdict
import random
import shutil

# Directory constants
RAW_IMG_DIR = "DATA/"

# Image extensions
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def rename_files(directory):
    """
    Renames files in the directory by replacing spaces and extra dots with underscores
    """
    for ext in EXTENSIONS:
        for file_path in glob.glob(os.path.join(directory, "**", ext), recursive=True):
            dir_name = os.path.dirname(file_path)
            base, ext = os.path.splitext(os.path.basename(file_path))

            # Replace spaces and extra dots
            new_base = base.replace(' ', '_').replace('.', '_')
            new_file_path = os.path.join(dir_name, new_base + ext.lower())

            # Rename if necessary
            if file_path != new_file_path:
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} → {new_file_path}")


def process_duplicates(directory, threshold=0):
    """
    Finds and removes duplicate images
    # Undouble reference: https://erdogant.github.io/undouble/pages/html/input_output.html
    """
    # Init undouble with default settings
    model = Undouble()

    # Import data
    model.import_data(directory)

    # Compute hashes
    model.compute_hash()

    # Find images with image-hash <= threshold
    model.group(threshold=threshold)

    # Plot duplicated images
    model.plot()

    # Paths for duplicate and non duplicate images
    duplicate_images = []
    non_duplicate_images = []

    # reference (line 119): https://github.com/erdogant/undouble/blob/main/undouble/examples.py
    # Extract pathnames for each group and classify them
    for idx_group in model.results['select_idx']:
        print(f"\nGroup {idx_group}:")

        group_images = [model.results['pathnames'][idx] for idx in idx_group]
        for img_path in group_images:
            print(f"- {img_path}")  # Shows the file paths for each group

        if len(idx_group) == 1:
            non_duplicate_images.append(group_images[0])
        else:
            keep_image = group_images[0]
            remove_images = group_images[1:]

            print(f"Keeping: {keep_image}")  # Print which image is kept

            for img_path in remove_images:
                try:
                    os.remove(img_path)
                    duplicate_images.append(img_path)
                    print(f"Removed duplicate: {img_path}")
                except FileNotFoundError:
                    print(f"File not found (could have already been moved): {img_path}")

    return duplicate_images, non_duplicate_images


# for reproducibility and consistent data split
random.seed(2025)

# Directory constants
RAW_IMG_DIR = "DATA/"
ALL_PROCESSED_DATA = "PROCESSED_DATA/"

VALIDATION_DATA = f'{ALL_PROCESSED_DATA}VALIDATION_DATA/'
TEST_DATA = f'{ALL_PROCESSED_DATA}TEST_DATA/'

TRAINING_DATA = f'{ALL_PROCESSED_DATA}TRAINING_DATA/'
AUGMENTED_OUTPUT_DIR = f'{TRAINING_DATA}TRAINING_AUGMENTED_DATA/'

# List constants
DIR_LIST = [ALL_PROCESSED_DATA, VALIDATION_DATA, TEST_DATA, \
            TRAINING_DATA, AUGMENTED_OUTPUT_DIR]

AUGMENTATION_LIST = ["original", "flipped_horizontally", "flipped_vertically", \
                     "rotated_90", "rotated_180", "brightness_enhanced"]


# ------- FUNCTIONS -------

def make_directories(dir_lst):
    '''
    Params: dir_lst - list of directories
    Function: makes
    '''
    for d in dir_lst:
        os.makedirs(d, exist_ok=True)
    print('Made necessary directories.')


def get_folders(directory):
    '''
    Param: directory - str
    Function: gets all subfolders in directory
    Returns: list of folder names
    '''
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folders.append(entry.name)
    return folders


def move_random_files(insect_source_dir, dest_dir, num_files):
    '''
    Param: source_dir - str, dest_dir - str, num_files - int
    Function: move specified number of files to desired folder
                choosing randomly (w/o replpacement)
    Returns: specified number of random files from source
                to destination directory
    '''

    # getting all files in this insect class
    files_in_source = []
    for f in os.listdir(insect_source_dir):
        if os.path.isfile(os.path.join(insect_source_dir, f)):
            files_in_source.append(f)

    # check there are enough to sample
    if len(files_in_source) < num_files:
        raise ValueError("Not enough files in the source directory")

    # there are enough so move files
    else:
        random_files = random.sample(files_in_source, num_files)

        # move the files
        for file in random_files:
            source_path = os.path.join(insect_source_dir, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.move(source_path, dest_path)

    return


def get_image_file(img_path):
    '''
    Param: img_path - str
    Function: Splits the string from last occurrence of forward slash.
    Returns: filename as str
    '''

    filename = img_path.rsplit('/', 1)[-1]
    return filename


# reference: https://stackoverflow.com/questions/44231209
def make_square_by_padding(image, min_size=224, fill_color='w'):
    '''
    Params: image - Image object, min_size - int, fill_color - 'w' or 'b'
    Function: pad rectangular images centrally to avoid stretching
            and comprimising integrity of image
    '''

    if fill_color == 'b':
        fill_color_code = (0, 0, 0, 0)

    elif fill_color == 'w':
        fill_color_code = (255, 255, 255, 0)
    else:
        raise ValueError('Incorrect fill color option: put w or b for a white or black background.')

    x, y = image.size

    # get largest dim so not stretching
    size = max(min_size, x, y)

    new_image = Image.new('RGBA', (size, size), fill_color_code)

    # centers image
    new_image.paste(image, (int((size - x) / 2), int((size - y) / 2)))

    return new_image


def resize_and_recolor(image_path, desired_square_dim=224, verbose=False):
    '''
    Param: image_path - str, desired_square_dim - int, verbose - boolean
    Function:
        # pad centrally to keep aspect ratio of rectangular images
        # convert to RBG format
        # standardize to desired size

    Returns: standardized image object (none if NA), log as dict of image errors if applicable
    '''

    error_log = defaultdict(list)
    img_name = get_image_file(image_path)

    try:

        image = Image.open(image_path)

        # pad to keep aspect ratio for model input
        image = make_square_by_padding(image)

        # convert to RGB - aka 3 channel
        image = image.convert("RGB")

        # Standardize to fixed image size
        image = image.resize((desired_square_dim, desired_square_dim))

        return image, error_log  # successful preprocessing (error log will be empty)

    except Exception as e:

        if verbose:
            print(f"Error processing {image_path}: {e}")

        # update error log
        error_log[img_name] = e

        return None, error_log  # unsuccessful preprocessing


def augment_image(img_object, output_subdir, base_filename, verbose=False):
    '''
    Param: img_object - Image object, output_subdir - str,
            base_filename - str, verbose - boolean
    Function:
        - create new images by flipping each horizontally,
        vertically, rotate, and brightness
        - saves augmented images in output folder

    Returns: boolean reflection image augmentation success, log as dict of image errors if applicable
    '''

    error_log = defaultdict(list)

    try:

        image = img_object

        # augmentation transformations to be applied
        augmentations = {
            "original": image,  # make sure to save the original image since it hasn't been saved yet!
            "flipped_horizontally": image.transpose(Image.FLIP_LEFT_RIGHT),
            "flipped_vertically": image.transpose(Image.FLIP_TOP_BOTTOM),
            "rotated_90": image.rotate(90),
            "rotated_180": image.rotate(180),
            "brightness_enhanced": ImageEnhance.Brightness(image).enhance(1.5)
        }

        # Apply each augmentation and save
        for aug_name, aug_image in augmentations.items():
            # save the 5 generated images AND the original (6 images total)
            aug_save_path = os.path.join(output_subdir, f"{base_filename}_{aug_name}.jpg")
            aug_image.save(aug_save_path)

        return True, error_log

    except Exception as e:

        if verbose:
            print(f"Error processing {base_filename}: {e}")

        # update error log
        error_log[base_filename] = e

        return False, error_log


def preprocess_data(class_folders_list, raw_data_directory, desired_output_directory):
    '''
    Params:
    Function: iterates through raw image files and does the following:
        - resizes
        - recolors
        - creates 5 image variations (augmentation)
    Returns: void since procedure

    '''

    # counters
    total_files = 0
    successfully_processed = 0

    # Process all insect class folders
    for folder in class_folders_list:

        # get the path of the folder
        folder_path = os.path.join(raw_data_directory, folder)

        # get path of output folder
        output_folder = os.path.join(desired_output_directory, folder)

        # make insect subfolders if DNE
        os.makedirs(output_folder, exist_ok=True)

        for root, subdirs, files in os.walk(folder_path):
            for file in files:  # for each image in RAW_IMG_DATA
                total_files += 1

                # get img files
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):

                    # get complete path to raw image
                    input_path = os.path.join(root, file)

                    # saving relative path to this img
                    relative_path = os.path.relpath(root, folder_path)

                    # constuct path to save processed img to corresponding output folder
                    output_subdir = os.path.join(output_folder, relative_path)

                    # saving info needed for output file saving
                    base_filename, file_extension = os.path.splitext(file)

                    # 1) standardize this img by resizing and recoloring
                    processed_img, process_error_log = resize_and_recolor(input_path)

                    # case: successfully processed img
                    if processed_img is not None:
                        processed_img_save_path = os.path.join(output_subdir, \
                                                               f"{base_filename}_processed.jpg")
                        processed_img.save(processed_img_save_path)
                        successfully_processed += 1

                else:
                    print(f"Skipped non-image file: {file}")

    # summary
    print(f"Successfully processed: {successfully_processed}, Total raw images: {total_files}")
    return


def normalize_image(img_object, range_type=1):
    '''
    Param: image - PIL Image object,
            range_type - int flag depicting type 1 range
            defined as [-1,1] or type 2 range defined as [0,1]
    Function: Normalized image as 3d NumPy array
    Returns: NumPy array
    '''
    image_array = np.array(img_object).astype(np.float32)

    if range_type == 1:
        normalized_array = (image_array / 127.5) - 1.0

    elif range_type == 2:
        normalized_array = (image_array - np.min(image_array)) / \
                           (np.max(image_array) - np.min(image_array))

    else:
        raise ValueError("Choose range_type as 1 for [-1,1] or 2 for [0,1]")

    return normalized_array


def split_data_into_train_valid_test(processed_data_folder_lst):
    '''
    Param: processed_data_folder_lst - list
    Function: Splits and moves all processed data (resizd and rgb coded images)
            into validation, test, and training sets
    Return: void since procedure
    '''

    for folder in processed_data_folder_lst:
        source_folder_path = os.path.join(ALL_PROCESSED_DATA, folder)

        # section out validation
        move_random_files(source_folder_path, VALIDATION_DATA, num_files=40)

        # section out test
        move_random_files(source_folder_path, TEST_DATA, num_files=20)

        # move the rest to training
        shutil.move(source_folder_path, TRAINING_DATA)

    print('Completed splitting data into training, validation, and test sets.')
    return


def perform_data_augmentation(training_folders_lst):
    '''
    Param: training_folders - list
    Function: reads throough training directory and augments each image,
              saving it to a new class specific augmented folder
    Return: void since procedure
    '''

    successfully_augmented = 0

    for folder in training_folders_lst:

        source_dir = os.path.join(TRAINING_DATA, folder)

        # set path for where to save augmented images
        output_subdir = os.path.join(AUGMENTED_OUTPUT_DIR, folder)
        os.makedirs(output_subdir, exist_ok=True)

        for file in os.listdir(source_dir):
            file_path = os.path.join(source_dir, file)

            # saving info needed for output file saving
            base_filename, file_extension = os.path.splitext(file)

            if os.path.isfile(file_path):
                # read image as Image object
                processed_img = Image.open(file_path)

                # perform and save augmentations
                final_img_success, augment_error_log = augment_image(processed_img, \
                                                                     output_subdir, \
                                                                     base_filename)

                if final_img_success == True:
                    successfully_augmented += 6

                else:
                    print(augment_error_log)

    print(f'Completed data augmentation on training set: training data now stands at {successfully_augmented} images')
    return


# Map of filename patterns to the correct class name
INSECT_CLASS_MAPPING = {
    "weevil": "adult_rice_weevil",
    "house_spider": "american_house_spider",
    "house_": "american_house_spider",
    "bedbug": "bedbug",
    "stink": "brown_stink_bug",
    "carpenterant": "carpenter_ant",
    "cellar": "cellar_spider",
    "flea": "flea",
    "centipede": "house_centipede",
    "silverfish": "silverfish",
    "termite": "subterranean_termite",
    "tick": "tick",
}


def organize_data(data_dir):
    """
    Create insect class folders and organize images to their correct folders
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist!")
        return

    # Create class folders if doesnt exist
    for class_name in INSECT_CLASS_MAPPING.values():
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Move images to their respective class folders
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)

        # Skip if it's already a directory
        if os.path.isdir(file_path):
            continue

        # Find correct class based on filename pattern
        for keyword, class_name in INSECT_CLASS_MAPPING.items():
            if keyword in file_name.lower():
                dest_path = os.path.join(data_dir, class_name, file_name)
                shutil.move(file_path, dest_path)
                print(f"Moved {file_name} → {dest_path}")
                break  # Stop checking once a match is found

def main():

    start = time.time()

    print("---------------START OF IMAGE PROCESSING---------------")

    # Rename files to avoid errors loading in image
    print("\nRenaming files---------------")
    rename_files(RAW_IMG_DIR)

    # Find and remove duplicate images
    print("\nProcessing duplicate images---------------")
    duplicate_images, non_duplicate_images = process_duplicates(RAW_IMG_DIR)

    print("\nHashing and removing duplicate images process complete.")
    print(f"Total duplicates removed: {len(duplicate_images)}")

    # 1) make directories that DNE
    print("---------------START OF MAKING DIRECTORIES AS NEEDED---------------")
    make_directories(DIR_LIST)

    # 2) get insect folders list
    print("---------------GETTING INSECT FOLDERS---------------")
    insect_folders = get_folders(RAW_IMG_DIR)

    # 3) resize and recolor data
    print("---------------PREPROCESSING RAW IMAGE DATA---------------")
    preprocess_data(insect_folders, RAW_IMG_DIR, ALL_PROCESSED_DATA)

    # 4) split data before augmentation to prevent data leakage
    print("---------------SPLITTING DATA (TRAIN/TEST/VALIDATION) BEFORE AUGMENTATION---------------")
    processed_insect_folders = get_folders(ALL_PROCESSED_DATA)
    processed_insect_folders = [f for f in processed_insect_folders if 'DATA' not in f]

    split_data_into_train_valid_test(processed_insect_folders)

    # 5) perform data augmentation on training set
    print("---------------PERFORMING DATA AUGMENTATION ON TRAINING SET---------------")
    training_folders = get_folders(TRAINING_DATA)
    training_folders = [f for f in processed_insect_folders if 'DATA' not in f]

    # 6) image normalization
    # can use function normalize_image(img_object, range_type = 1) BUT
    # holding off on applying since transfer learning models have built in
    # normalization layers

    perform_data_augmentation(training_folders)

    print("---------------SORTING FILES TO FINAL DESETINATIONS---------------")
    PROCESSED_DATA = "PROCESSED_DATA"
    TEST_DATA = os.path.join(PROCESSED_DATA, "TEST_DATA")
    VALIDATION_DATA = os.path.join(PROCESSED_DATA, "VALIDATION_DATA")

    # Organize the data in the TEST_DATA and VALIDATION_DATA folders
    organize_data(TEST_DATA)
    organize_data(VALIDATION_DATA)

    print("TEST_DATA and VALIDATION_DATA have been organized into class folders.")

    end = time.time()
    print(f'Processing pipeline complete - took approximately {round(end-start)} seconds')



if __name__ == "__main__":
    main()
