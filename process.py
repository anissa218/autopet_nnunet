import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

print('imports successful')

# fabian's script

class Autopet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces

    def predict(self):
        """
        Your algorithm goes here
        """
        print("nnUNet segmentation starting!")

        os.environ['nnUNet_compile'] = 'F'  # on my system the T does the test image in 2m56 and F in 3m15. Not sure if
        # 20s is worth the risk

        maybe_mkdir_p(self.output_path)

        trained_model_path = 'nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs_NoMirroring__nnUNetPlans__3d_fullres' # anissa changes

        ct_mha = subfiles(join(self.input_path, 'images/ct/'), suffix='.mha')[0]
        pet_mha = subfiles(join(self.input_path, 'images/pet/'), suffix='.mha')[0]
        uuid = os.path.basename(os.path.splitext(ct_mha)[0])
        output_file_trunc = os.path.join(self.output_path, uuid)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(trained_model_path, use_folds=(0, ),checkpoint_name='checkpoint_best.pth') # anissa changed from this use_folds=(0, 1, 2, 3, 4)
        predictor.dataset_json['file_ending'] = '.mha'

        # ideally we would like to use predictor.predict_from_files but this stupid docker container will be called
        # for each individual test case so that this doesn't make sense
        images, properties = SimpleITKIO().read_images([ct_mha, pet_mha])
        predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)

        print('Prediction finished')

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        # self.check_gpu() # waste of time. 10 mins is tight, yo
        print('Start prediction')
        self.predict()
        print('done')


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
