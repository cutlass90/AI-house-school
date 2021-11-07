import os
import random
import string
from glob import glob
from os.path import join, basename
from shutil import copyfile
from tqdm import tqdm


def create_solutioncsv(path2_test_dir):
    paths = glob(join(path2_test_dir, '*.mp4'))
    random.shuffle(paths)

    with open('solution.csv', 'w') as f:
        f.writelines(['filename,label,Usage\n'])

    for path in paths:
        with open('solution.csv', 'a') as f:
            f.writelines(
                [f'{basename(path)},{random.randint(0, 1)},{"Public" if random.random() > 0.5 else "Private"}\n'])


def crete_sample_submission(path2_test_dir):
    paths = glob(join(path2_test_dir, '*.mp4'))
    random.shuffle(paths)
    with open('sample_submission.csv', 'w') as f:
        f.writelines(['filename,label\n'])

    for path in paths:
        with open('sample_submission.csv', 'a') as f:
            f.writelines([f'{basename(path)},{random.randint(0, 1)}\n'])


def collect_and_rename():
    os.makedirs('dataset', exist_ok=True)
    paths = glob('/home/nazar/DATASETS/videos/LRS2/main/*/*.mp4') + glob(
        '/home/nazar/DATASETS/videos/reenactment_vox/vox_train/*.mp4')
    print(f'{len(paths)} files were found')
    for path in paths:
        new_file_name = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
        copyfile(path, f'dataset/{new_file_name}.mp4')



def form_dataset(path_to_data, test_part=0.3):
    os.makedirs(join(path_to_data, 'train'), exist_ok=True)
    os.makedirs(join(path_to_data, 'test'), exist_ok=True)
    with open('train.csv', 'w') as f:
        f.writelines(['filename,label\n'])

    with open('solution.csv', 'w') as f:
        f.writelines(['filename,label,Usage\n'])

    with open('file_mapping.csv', 'w') as f:
        f.writelines(['filename,type\n'])

    paths = {}
    paths['lipsync'] = glob(join(path_to_data, 'lipsynced/*.mp4'))
    paths['reenactment'] = glob(join(path_to_data, 'reenacted/*.mp4'))
    paths['swap'] = glob(join(path_to_data, 'swapped/*.mp4'))
    paths['original'] = glob(join(path_to_data, 'originals/*.mp4'))

    for video_type, paths in paths.items():
        random.shuffle(paths)
        for path in tqdm(paths):
            new_file_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(32)) + '.mp4'
            sub_dir = "train" if random.random() > test_part else "test"
            dst_path = join(path_to_data, sub_dir, new_file_name)
            copyfile(path, dst_path)

            if "train" == sub_dir:
                with open('train.csv', 'a') as f:
                    f.writelines([f'{new_file_name},{0 if video_type == "original" else 1}\n'])
            elif "test" == sub_dir:
                with open('solution.csv', 'a') as f:
                    f.writelines(
                        [f'{new_file_name},{0 if video_type == "original" else 1},{"Public" if random.random() > 0.5 else "Private"}\n'])
            else:
                raise ValueError

            with open('file_mapping.csv', 'a') as f:
                f.writelines([f'{new_file_name},{video_type}\n'])

            print()







def main():
    # form_dataset('/home/nazar/datasets/reface_kaggle_v1')
    crete_sample_submission('dataset/test')

if __name__ == "__main__":
    main()
