import os, pickle
import pandas as pd


def whether_in_the_list(name, epochs):
    # false to skip (continue), true to do analysis
    for epoch in epochs:
        if f"_{epoch}." in f"_{name}.":
            return True
        if epoch == "best" and name =="best":
            return True
    return False