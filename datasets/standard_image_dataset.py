from torch.utils.data import DataLoader, Dataset

def get_data_loader(
    img_list,
    batch_size=20,
    num_workers=0,
):
    """
    Create the dataloader for the train dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    
    train_set = ImageDataset(
        img_list= img_list
    )
    data_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return data_loader

class ImageDataset(Dataset):
    def __init__(
        self,
        img_list,
    ):
        """
        Init function for the ETH-XGaze dataset, create key pairs to shuffle the dataset.

        :dataset_path: Path to the subjects files with all the information
        :keys_to_use: The subjects ID to use for the dataset
        :sub_folder: Indicate if it has to create the train,validation or test dataset
        :num_val_images: Used only for the validation dataset, indicate how many images to include in the validation dataset
        :transform: All the transformations to apply to the images
        :is_shuffle: Boolean value that indicates if images will be shuffled
        :index_file: Path to a specific key pairs file
        """

        self.images = img_list

    def __len__(self):
        """
        Function that returns the length of the dataset.

        :return: Returns the length of the dataset
        """

        return len(self.images)

    def __del__(self):
        """
        Close all the hdfs files of the subjects.

        """
        del self.images


    def __getitem__(self, idx):
        
        return { 'images' : self.images[idx] }