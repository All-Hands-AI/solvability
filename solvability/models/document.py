from __future__ import annotations

import pandas as pd
from analysis.models.data import Data
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


class Document(BaseModel):
    issue_id: str
    issue_description: str
    resolved: bool


class DocumentSplit(BaseModel):
    train: list[Document]
    test: list[Document]
    validation: list[Document]

    @staticmethod
    def initialize(
        data: Data, evaluation: str, test_size: float = 0.2, validation_size: float = 0.2, random_seed: int = 8675309
    ) -> DocumentSplit:
        """Initialize a DocumentSplit object with train, test, and validation sets.

        Args:
            data: Data object containing the dataset
            evaluation: Name of the system providing the ground truth labels
            test_size: Proportion of the dataset to include in the test split
            validation_size: Proportion of the dataset to include in the validation split
            random_seed: Random seed for reproducibility
        """

        # Split the dataset into train and test sets
        train_data, test_and_validation_data = train_test_split(
            data.dataset.instances,
            test_size=test_size + validation_size,
            random_state=random_seed,
            stratify=[
                data.systems[evaluation].results.is_resolved(instance.instance_id)
                for instance in data.dataset.instances
            ],
        )

        # Further split the test set into validation and test sets
        test_data, validation_data = train_test_split(
            test_and_validation_data,
            test_size=validation_size / (test_size + validation_size),
            random_state=random_seed,
            stratify=[
                data.systems[evaluation].results.is_resolved(instance.instance_id)
                for instance in test_and_validation_data
            ],
        )

        # Create Document objects for each split
        train_documents = [
            Document(
                issue_id=instance.instance_id,
                issue_description=instance.problem_statement,
                resolved=data.systems[evaluation].results.is_resolved(instance.instance_id),
            )
            for instance in train_data
        ]
        test_documents = [
            Document(
                issue_id=instance.instance_id,
                issue_description=instance.problem_statement,
                resolved=data.systems[evaluation].results.is_resolved(instance.instance_id),
            )
            for instance in test_data
        ]
        validation_documents = [
            Document(
                issue_id=instance.instance_id,
                issue_description=instance.problem_statement,
                resolved=data.systems[evaluation].results.is_resolved(instance.instance_id),
            )
            for instance in validation_data
        ]

        return DocumentSplit(train=train_documents, test=test_documents, validation=validation_documents)

    def to_df(self) -> pd.DataFrame:
        """Convert the DocumentSplit object to a DataFrame."""
        rows = []
        for doc in self.train:
            rows.append(
                {
                    "issue_id": doc.issue_id,
                    "issue_description": doc.issue_description,
                    "resolved": doc.resolved,
                    "split": "train",
                }
            )
        for doc in self.test:
            rows.append(
                {
                    "issue_id": doc.issue_id,
                    "issue_description": doc.issue_description,
                    "resolved": doc.resolved,
                    "split": "test",
                }
            )
        for doc in self.validation:
            rows.append(
                {
                    "issue_id": doc.issue_id,
                    "issue_description": doc.issue_description,
                    "resolved": doc.resolved,
                    "split": "validation",
                }
            )
        return pd.DataFrame(rows)
