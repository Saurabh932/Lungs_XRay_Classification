import os
import sys
from typing import Tuple

import joblib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from Xray.entity.artifact_entity import DataIngestionArtifact 
from Xray.entity.artifact_entity import DataTransformationArtifact
from Xray.entity.config_entity import DataTransformationConfig
from Xray.exception import XRayException
from Xray.logger import logging

class DataTransformation:
    def __init__(self, 
                data_ingestion_artifact: DataIngestionArtifact,
                data_transformation_config: DataTransformationConfig
                ):
        
        self.data_ingestion_artifcat = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config


    def transforming_training_data(self) -> transforms.Compose:
        try:
            logging.info("Entered the transforming_training_data method of Data transformation class")

            train_transform : transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ColorJitter(**self.data_transformation_config.color_jitter_transforms),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(self.data_transformation_config.RANDOMROTATION),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.data_transformation_config.normalize_transforms)
                ]
            )
            logging.info("Exited the transforming_training_data method of Data transformation class")

            return train_transform

        except Exception as e:
            raise XRayException(e, sys)
        
    
    def transforming_testing_data(self) -> transforms.Compose:
        logging.info("Entered the transforming_testing_data method of Data Transformation.")
        try:
            test_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.data_transformation_config.normalize_transforms)
                ]
            )
            logging.info("Exited the transforming_testing_data method of Data transformation class")

            return test_transform

        except Exception as e:
            raise XRayException(e, sys)
        
    
    def data_loader(self, train_transform: transforms.Compose, test_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
        try:
            logging.info("Enterd the data_loader method of Data Transformation class")

            train_data: Dataset = ImageFolder(
                                    os.path.join(self.data_ingestion_artifcat.train_file_path),
                                    transform=train_transform
                                    )
            test_data: Dataset = ImageFolder(
                                    os.path.join(self.data_ingestion_artifcat.test_file_path),
                                    transform=test_transform,
                                    )

            logging.info("Created train data and test data paths")

            train_loader: DataLoader = DataLoader(
                                        train_data, **self.data_transformation_config.data_loader_params
                                        )
            
            test_loader: DataLoader = DataLoader(
                                        test_data, **self.data_transformation_config.data_loader_params
                                        )
            
            logging.info("Exited the data_loader method of Data Transformation class")

            return train_loader, test_loader
        
        except Exception as e:
            raise XRayException(e,sys)
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Enter the initiate_data_transformation method of data transformation class")

            train_transform: transforms.Compose = self.transforming_training_data()
            test_transform: transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(train_transform, self.data_transformation_config.train_transforms_file)
            joblib.dump(test_transform, self.data_transformation_config.test_transforms_file)

            train_loader, test_loader = self.data_loader(
                                        train_transform = train_transform,
                                        test_transform = test_transform
                                        )

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                                                transformed_train_object = train_loader,
                                                transformed_test_object = test_loader,
                                                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                                                test_transform_file_path=self.data_transformation_config.test_transforms_file,

                                                )
            
            logging.info(
                "Exited the initiate_data_transformation method of Data transformation class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)