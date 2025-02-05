"""Base class to standardize the behavior of all (sub)modules."""

# Author: Max Pinheiro Jr <maxjr82@gmail.com>
# Date: June 15 2023

import io

import numpy as np
import pandas as pd


class BaseClass:
    """Provide a list of class name and type for each attribute."""
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        attributes = ", ".join(
            f"{k}={type(v)}" for k, v in self.__dict__.items()
        )
        return f"{class_name}({attributes})"

    def __str__(self) -> str:
        """Provide a summary of the current class status.

        :return: Short description of the class state with attributes and
                 values.
        :rtype: str
        """
        # Select all variables belonging to the class
        vars = [
            attr
            for attr in self.__dir__()
            if not callable(getattr(self, attr))
        ]
        vars = [attr for attr in vars if not attr.startswith("__")]
        class_name = self.__class__.__name__
        cls_status = "-" * (len(class_name) + 39) + "\n"
        cls_status += f"Current status of the {class_name} class variables:\n"
        cls_status += "-" * (len(class_name) + 39) + "\n\n"
        # Loop over the class attributes to add the respective info state
        # to the str
        for name in vars:
            attr = getattr(self, name)
            if name == "labels":
                attr = list(attr)
            if isinstance(attr, np.ndarray):
                cls_status += (
                    f"  \u2022 {name}, array of shape -> {attr.shape}\n"
                )
            elif isinstance(attr, pd.DataFrame):
                cls_status += f"  \u2022 {name} shape -> {attr.shape}\n\n"
                buf = io.StringIO()
                attr.info(buf=buf)
                data_info = buf.getvalue()
                cls_status += data_info + "\n"
            else:
                cls_status += f"  \u2022 {name} -> {attr}\n"
        return cls_status
