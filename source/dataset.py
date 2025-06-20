import os
import re
import logging
import random
from dataclasses import dataclass
from glob import glob

import extract_msg
import torch
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from tqdm import tqdm

from configurations.params import DATASET_PATH


logger = logging.getLogger(__name__)

extract_msg.msg_classes.message_base.logger.setLevel(level=logging.INFO)
extract_msg.msg_classes.msg.logger.setLevel(level=logging.WARNING)


@dataclass
class EmailElem:
    email_cls: str
    email_id: str
    email_text: str
    email_filename: str


class EmailDataset(Dataset):

    def __init__(self,
                 dataset_path: str | None = DATASET_PATH,
                 cache: bool = True,
                 extention: str = "msg",
                 limit_number_of_samples: int = 20,
                 ):
        super().__init__()

        self.path = dataset_path

        self._all_files = (glob(f"{self.path}/*/*.{extention}") if self.path is not None else [])
        self._all_files.sort()

        files_by_class: dict[str, list[str]] = {}
        for file_path in self._all_files:
            file_class = os.path.dirname(file_path)
            if file_class in files_by_class:
                files_by_class[file_class].append(file_path)
            else:
                files_by_class[file_class] = [file_path]

        
        self._all_files = []
        for key, value in files_by_class.items():
            files_by_class[key] = value[:limit_number_of_samples]
            self._all_files.extend(value[:limit_number_of_samples])

        self.data: list[tuple[str, str]] | None = None
        self.email_classes: dict[str, str] = {}

        if cache:
            self.data: list[EmailElem] = []
            self.dict_data: dict[str, list[EmailElem]] = {}
            
            for file_path in tqdm(self._all_files, desc="Loading dataset..."):
                email_text, email_cls = self.load_file(file_path=file_path)
                if email_cls not in self.email_classes:
                    self.email_classes[email_cls] = (email_cls)

                self.data.append(
                    EmailElem(
                        email_cls=email_cls,
                        email_id=(
                            f"{email_cls}{len(self.dict_data[email_cls])}" 
                            if email_cls in self.dict_data else f"{email_cls}0"
                        )
                    )
                )

                if email_cls not in self.dict_data:
                    self.dict_data[email_cls] = [EmailElem(
                        email_cls=email_cls,
                        email_id=f"{email_cls}0",
                        email_text=email_text,
                        email_filename=file_path,
                    )]
                else:
                    self.dict_data[email_cls].append(EmailElem(
                        email_cls=email_cls,
                        email_id=f"{email_cls}{len(self.dict_data[email_cls])}",
                        email_text=email_text,
                        email_filename=file_path,
                    ))


        assert len(self._all_files) == len(self.data), f"length of filenames does not match number of loaded files {len(self._all_files) != len(self.data)}"

    
    def load_file_text(self, file_path: str) -> tuple[str]:
        with open(file_path, encoding="utf-8", mode="r") as file:
            file_content = file.read()

        file_name = os.path.basename(file_path)
        item_cls = ("_").join(file_name.split("_")[:-1])

        return file_content, item_cls
    

    def load_file(self, file_path: str) -> tuple[str, str]:
        """ 
        Load the .msg file and its class

        Args:
            file_path: the msg file path

        Returns:
            loaded text
            class
        """

        msg = extract_msg.openMsg(path=file_path, html=True)
        soup = BeautifulSoup(msg.htmlBody, features="html.parser")

        all_email_table_headers = ""
        for table in soup.find_all("table"):
            headers = [th.text.strip() for th in table.find_all("th")]
            headers = (", ").join(headers)
            all_email_table_headers += f"{headers}\n"

        
        for script in soup(["script", "style", "table"]):
            script.extract()

        file_content = soup.get_text()
        file_content = re.sub(f"[^\S\n\t]+", " ", file_content)
        file_content = re.sub(r"(\n)(\1+)", " ", file_content)
        file_content = file_content.replace("\n ", "\n")

        if len(file_content) > 0 and file_content[0] == " ":
            file_content = file_content[1:]
        
        file_content = (f"subject: {msg.subject}\n{file_content}\n{all_email_table_headers}")
        
        item_cls = os.path.basename(os.path.dirname(file_path))

        return file_content, item_cls
    

    @classmethod
    def from_dict(cls: "EmailDataset", dataset_dict: dict[str, list[str]]) -> "EmailDataset":
        email_dataset = EmailDataset(None)
        email_dataset.dict_data = dataset_dict

        for key, value in list(dataset_dict.items()):
            email_dataset.data.extend([val for val in value])
            email_dataset.email_classes[key] = key
        
        email_dataset._all_files = [elem.email_filename for elem in email_dataset.data]

        return email_dataset
    

    def shuffle(self):
        if self.data is not None:
            random.shuffle(self.data)
        else:
            random.shuffle(self._all_files)


    def __len__(self) -> int:
        return len(self._all_files)
    

    def __getitem__(self, index: int) -> EmailElem:
        if self.data is not None:
            return self.data[index]
        
        email_text, email_cls = self.load_file(self._all_files[index])

        return EmailElem(
            email_cls=email_cls,
            email_id=None,
            email_text=email_text,
            email_filename=self._all_files[index],
        )
    

def split_dataset(ratio: float, dataset: EmailDataset) -> tuple[EmailDataset, EmailDataset]:
    class_length = len(dataset.dict_data[list(dataset.dict_data.keys())[0]])
    n_samples = int(ratio * class_length)
    indices = random.sample(range(0, class_length), n_samples)
    negative_indices = [index for index in range(0, class_length) if index not in indices]

    val_d, test_d = {}, {}
    for key, value in dataset.dict_data.items():
        val_d[key] = [value[i] for i in indices]
        test_d[key] = [value[i] for i in negative_indices]

    return EmailDataset.from_dict(val_d), EmailDataset.from_dict(test_d)


class FinetuningDataset(torch.utils.data.Dataset):

    def __init__(self, email_dataset: EmailDataset):
        super().__init__()

        email_dataset.shuffle()
        self.email_dataset = email_dataset
    
    def __len__(self) -> int:
        return len(self.email_dataset) // 2 + (len(self.email_dataset) % 2)
    
    def __getitem__(self, index: int) -> EmailElem:
        return self.email_dataset.__getitem__(index), self.email_dataset.__getitem__(index + 1)
    

class TripleDataset(torch.utils.data.Dataset):
    """ The dataset returns an anchor, positive and negative email pair."""

    def __init__(self, email_dataset: EmailDataset):
        super().__init__()
        
        email_dataset.shuffle()
        self.email_dataset = email_dataset
        self.num_email_classes = len(list(email_dataset.email_classes.keys()))

    def __len__(self) -> int:
        return len(self.email_dataset)
    
    def __getitem__(self, index: int):
        anchor_class_index = index % self.num_email_classes
        anchor_class = list(self.email_dataset.email_classes.values())[anchor_class_index]

        number_class_samples = len(self.email_dataset.dict_data[anchor_class])
        
        anchor_index = index % number_class_samples
        
        positive_index = [i for i in range(number_class_samples) if i != anchor_index[random.randint(0, number_class_samples-2)]]

        negative_class_index = [i for i in range(self.num_email_classes) if i != anchor_class_index[random.randint(0, self.num_email_classes-2)]]
        negative_class = list(self.email_dataset.email_classes.values())[negative_class_index]

        negative_index = random.randint(0, len(self.email_dataset.dict_data[negative_class]) - 1)

        return (
            self.email_dataset.dict_data[anchor_class][anchor_index],
            self.email_dataset.dict_data[anchor_class][positive_index],
            self.email_dataset.dict_data[negative_class][negative_index],
        )